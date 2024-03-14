/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/functions/sparksql/DecimalUtil.h"

namespace facebook::velox::functions::aggregate::sparksql {

static TypePtr getSumType(const TypePtr& rawInputType) {
  if (rawInputType->isRow()) {
    return rawInputType->childAt(0);
  }
  VELOX_USER_CHECK(rawInputType->isDecimal());
  auto [p, s] = getDecimalPrecisionScale(*rawInputType.get());
  // In Spark,
  // sum precision = input precision + 10
  // sum scale = input scale
  return DECIMAL(std::min<uint8_t>(LongDecimalType::kMaxPrecision, p + 10), s);
}

static TypePtr getAvgType(const TypePtr& rawInputType) {
  VELOX_USER_CHECK(rawInputType->isDecimal());
  auto [p, s] = getDecimalPrecisionScale(*rawInputType.get());
  // In Spark,
  // sum precision = input precision + 4
  // sum scale = input scale + 4
  return DECIMAL(
      std::min<uint8_t>(LongDecimalType::kMaxPrecision, p + 4),
      std::min<uint8_t>(LongDecimalType::kMaxPrecision, s + 4));
}

/// @tparam TInputType The raw input data type.
/// @tparam TResultType The type of the output average value.
template <typename TInputType, typename TResultType>
class DecimalAverageAggregate {
 public:
  using InputType = Row<TInputType>;

  using IntermediateType =
      Row</*sum*/ int128_t,
          /*count*/ int64_t>;

  using OutputType = TResultType;

  struct FunctionState {
    TypePtr resultType;
    TypePtr sumType;
  };

  /// Spark's decimal sum doesn't have the concept of a null group, each group
  /// is initialized with an initial value, where sum = 0 and count = 0. The
  /// final agg may fallback to being executed in Spark, so the meaning of the
  /// intermediate data should be consistent with Spark. Therefore, we need to
  /// use the parameter nonNullGroup in writeIntermediateResult to output a null
  /// group as sum = 0, count = 0. nonNullGroup is only available when
  /// default-null behavior is disabled.
  static constexpr bool default_null_behavior_ = false;

  static constexpr bool aligned_accumulator_ = true;

  static void initialize(
      FunctionState& state,
      const std::vector<TypePtr>& rawInputTypes,
      const TypePtr& resultType,
      const std::vector<VectorPtr>& constantInputs) {
    state.resultType = resultType;
    VELOX_CHECK_LE(rawInputTypes.size(), 1);
    auto rawInputType = rawInputTypes[0];
    state.sumType = getSumType(rawInputType);
  }

  static bool toIntermediate(
      exec::out_type<Row<int128_t, int64_t>>& out,
      exec::optional_arg_type<TInputType> in) {
    if (in.has_value()) {
      out.copy_from(std::make_tuple(in.value(), 1));
    } else {
      out.copy_from(std::make_tuple(0, 0));
    }
    return true;
  }

  /// This struct stores the sum of input values, overflow during accumulation,
  /// and the count number of the input values. If the count is not 0, then if
  /// sum is nullopt that means an overflow has happened.
  struct AccumulatorType {
    std::optional<int128_t> sum{0};
    int64_t overflow{0};
    int64_t count{0};

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* /*allocator*/,
        const FunctionState& /*state*/) {}

    std::optional<int128_t> computeFinalResult(
        const FunctionState& state) const {
      if (!sum.has_value()) {
        return std::nullopt;
      }
      auto const adjustedSum =
          DecimalUtil::adjustSumForOverflow(sum.value(), overflow);
      if (!adjustedSum.has_value()) {
        // Found overflow during computing adjusted sum.
        return std::nullopt;
      }

      VELOX_USER_CHECK(
          state.resultType->isDecimal(),
          "resultType must be decimal type, but found");
      auto [resultPrecision, resultScale] =
          getDecimalPrecisionScale(*state.resultType.get());
      VELOX_USER_CHECK(state.sumType->isDecimal());
      auto [sumPrecision, sumScale] =
          getDecimalPrecisionScale(*state.sumType.get());

      // Spark use DECIMAL(20,0) to represent long value.
      const uint8_t countPrecision = 20, countScale = 0;

      auto [dividePrecision, divideScale] =
          functions::sparksql::DecimalUtil::dividePrecisionScale(
              sumPrecision, sumScale, countPrecision, countScale);

      auto sumRescale = divideScale - sumScale + countScale;
      int128_t avg;
      bool overflow = false;
      functions::sparksql::DecimalUtil::
          divideWithRoundUp<int128_t, int128_t, int128_t>(
              avg, adjustedSum.value(), count, sumRescale, overflow);
      if (overflow) {
        return std::nullopt;
      } else {
        TResultType rescaledValue;
        const auto status =
            DecimalUtil::rescaleWithRoundUp<int128_t, TResultType>(
                avg,
                dividePrecision,
                divideScale,
                resultPrecision,
                resultScale,
                rescaledValue);
        return status.ok() ? std::optional<int128_t>(rescaledValue)
                           : std::nullopt;
      }
    }

    bool addInput(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<TInputType> data,
        const FunctionState& /*state*/) {
      if (!data.has_value()) {
        return false;
      }
      if (!sum.has_value()) {
        // sum is initialized to 0. When it is nullopt, it implies that the
        // count number of the input values must not be 0.
        VELOX_CHECK(count != 0)
        return true;
      }
      int128_t result;
      overflow +=
          DecimalUtil::addWithOverflow(result, data.value(), sum.value());
      sum = result;
      count += 1;
      return true;
    }

    bool combine(
        HashStringAllocator* /*allocator*/,
        exec::optional_arg_type<Row<int128_t, int64_t>> other,
        const FunctionState& /*state*/) {
      if (!other.has_value()) {
        return false;
      }
      auto const otherSum = other.value().template at<0>();
      auto const otherCount = other.value().template at<1>();

      // otherCount is never null.
      VELOX_CHECK(otherCount.has_value());
      if (count == 0 && otherCount.value() == 0) {
        // Both accumulators have no input values, no need to do the
        // combination.
        return false;
      }

      bool currentOverflow = count > 0 && !sum.has_value();
      bool otherOverflow = otherCount.value() > 0 && !otherSum.has_value();
      if (currentOverflow || otherOverflow) {
        sum = std::nullopt;
        count += otherCount.value();
      } else {
        int128_t result;
        overflow +=
            DecimalUtil::addWithOverflow(result, otherSum.value(), sum.value());
        sum = result;
        count += otherCount.value();
      }
      return true;
    }

    bool writeIntermediateResult(
        bool nonNullGroup,
        exec::out_type<IntermediateType>& out,
        const FunctionState& /*state*/) {
      if (!nonNullGroup) {
        // If a group is null, all values in this group are null. In Spark, this
        // group will be the initial value, where sum is 0 and count is 0.
        out = std::make_tuple(static_cast<int128_t>(0), 0L);
      } else {
        if (!sum.has_value()) {
          // Sum should be set to null on overflow.
          out.template set_null_at<0>();
          out.template get_writer_at<1>() = count;
        } else {
          auto adjustedSum =
              DecimalUtil::adjustSumForOverflow(sum.value(), overflow);
          if (adjustedSum.has_value()) {
            out = std::make_tuple(adjustedSum.value(), count);
          } else {
            out.template set_null_at<0>();
            out.template get_writer_at<1>() = count;
          }
        }
      }
      return true;
    }

    bool writeFinalResult(
        bool nonNullGroup,
        exec::out_type<OutputType>& out,
        const FunctionState& state) {
      if (!nonNullGroup || count == 0) {
        // In Spark, if all inputs are null, count will be 0, and the result of
        // average value will be null.
        return false;
      }
      auto finalResult = computeFinalResult(state);
      if (finalResult.has_value()) {
        out = static_cast<TResultType>(finalResult.value());
        return true;
      } else {
        // Sum should be set to null on overflow.
        return false;
      }
    }
  };
};

} // namespace facebook::velox::functions::aggregate::sparksql
