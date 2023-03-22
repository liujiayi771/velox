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

#include "velox/common/base/IOUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/DecimalAggregate.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql::aggregates {

using velox::aggregate::LongDecimalWithOverflowState;

template <typename TInputType, typename TResultType>
class DecimalAverageAggregate : public exec::Aggregate {
 public:
  explicit DecimalAverageAggregate(TypePtr inputType, TypePtr resultType)
      : exec::Aggregate(resultType), inputType_(inputType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalAverageAggregate);
  }

  int32_t accumulatorAlignmentSize() const override {
    return sizeof(LongDecimalWithOverflowState);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_)
          velox::aggregate::LongDecimalWithOverflowState();
    }
  }

  void addRawInput(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.applyToSelected([&](vector_size_t i) {
          updateNonNullValue(groups[i], UnscaledLongDecimal(value));
        });
      } else {
        // Spark expects the result of partial avg to be non-nullable.
        rows.applyToSelected(
            [&](vector_size_t i) { exec::Aggregate::clearNull(groups[i]); });
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        // Spark expects the result of partial avg to be non-nullable.
        exec::Aggregate::clearNull(groups[i]);
        if (decodedRaw_.isNullAt(i)) {
          return;
        }
        updateNonNullValue(
            groups[i], UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)));
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], UnscaledLongDecimal(data[i]));
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(
            groups[i], UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)));
      });
    }
  }

  void addSingleGroupRawInput(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /*mayPushdown*/) override {
    decodedRaw_.decode(*args[0], rows);
    if (decodedRaw_.isConstantMapping()) {
      if (!decodedRaw_.isNullAt(0)) {
        const auto numRows = rows.countSelected();
        int64_t overflow = 0;
        int128_t totalSum{0};
        auto value = decodedRaw_.valueAt<TInputType>(0);
        rows.template applyToSelected([&](vector_size_t i) {
          updateNonNullValue(group, UnscaledLongDecimal(value));
        });
      } else {
        // Spark expects the result of partial avg to be non-nullable.
        exec::Aggregate::clearNull(group);
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              group, UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)));
        } else {
          // Spark expects the result of partial avg to be non-nullable.
          exec::Aggregate::clearNull(group);
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      const TInputType* data = decodedRaw_.data<TInputType>();
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum, data[i].unscaledValue(), accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators<false>(group, serialized);
    } else {
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        accumulator.overflow += DecimalUtil::addWithOverflow(
            accumulator.sum,
            decodedRaw_.valueAt<TInputType>(i).unscaledValue(),
            accumulator.sum);
      });
      accumulator.count = rows.countSelected();
      char rawData[LongDecimalWithOverflowState::serializedSize()];
      StringView serialized(
          rawData, LongDecimalWithOverflowState::serializedSize());
      accumulator.serialize(serialized);
      mergeAccumulators(group, serialized);
    }
  }

  void addIntermediateResults(
      char** groups,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumCol = baseRowVector->childAt(0);
    auto countCol = baseRowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        addIntermediateDecimalResults(
            groups,
            rows,
            sumCol->as<SimpleVector<UnscaledShortDecimal>>(),
            countCol->as<SimpleVector<int64_t>>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        addIntermediateDecimalResults(
            groups,
            rows,
            sumCol->as<SimpleVector<UnscaledLongDecimal>>(),
            countCol->as<SimpleVector<int64_t>>());
        break;
      }
      default:
        VELOX_FAIL(
            "Unsupported sum type for decimal aggregation: {}",
            sumCol->typeKind());
    }
  }

  template <class UnscaledType>
  void addIntermediateDecimalResults(
      char** groups,
      const SelectivityVector& rows,
      SimpleVector<UnscaledType>* sumVector,
      SimpleVector<int64_t>* countVector) {
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          auto accumulator = decimalAccumulator(groups[i]);
          mergeSumCount(accumulator, sum, count);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        mergeSumCount(accumulator, sum, count);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        mergeSumCount(accumulator, sum, count);
      });
    }
  }

  void addSingleGroupIntermediateResults(
      char* group,
      const SelectivityVector& rows,
      const std::vector<VectorPtr>& args,
      bool /* mayPushdown */) override {
    decodedPartial_.decode(*args[0], rows);
    auto baseRowVector = dynamic_cast<const RowVector*>(decodedPartial_.base());
    auto sumVector = baseRowVector->childAt(0)->as<SimpleVector<TInputType>>();
    auto countVector = baseRowVector->childAt(1)->as<SimpleVector<int64_t>>();

    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected(
            [&](vector_size_t i) { mergeAccumulators(group, sum, count); });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, count);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto count = countVector->valueAt(decodedIndex);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, count);
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumVector = rowVector->childAt(0)->asFlatVector<TResultType>();
    auto countVector = rowVector->childAt(1)->asFlatVector<int64_t>();
    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);

    uint64_t* rawNulls = getRawNulls(rowVector);

    int64_t* rawCounts = countVector->mutableRawValues();
    TResultType* rawSums = sumVector->mutableRawValues();

    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* accumulator = decimalAccumulator(group);
        rawCounts[i] = accumulator->count;
        rawSums[i] = (TResultType)accumulator->sum;
      }
    }
  }

  TResultType computeFinalValue(LongDecimalWithOverflowState* accumulator) {
    // Handles round-up of fraction results.
    auto [sumPrecision, sumScale] =
        getDecimalPrecisionScale(*this->inputType().get());
    auto [rPrecision, rScale] =
        getDecimalPrecisionScale(*this->resultType().get());
    int countScale = 0;
    auto sumRescale = computeRescaleFactor(sumScale, countScale, rScale);

    TResultType average = TResultType(0);
    if constexpr (std::is_same_v<TInputType, UnscaledLongDecimal>) {
      if constexpr (std::is_same_v<TResultType, UnscaledLongDecimal>) {
        // Spark use DECIMAL(20, 0) to represent long value
        auto countDecimal = UnscaledLongDecimal(accumulator->count);

        DecimalUtil::divideWithRoundUp<
            UnscaledLongDecimal,
            UnscaledLongDecimal,
            UnscaledLongDecimal>(
            average,
            (UnscaledLongDecimal)accumulator->sum,
            countDecimal,
            false,
            sumRescale,
            0);
      } else if constexpr (std::is_same_v<TResultType, UnscaledShortDecimal>) {
        // we enter this case when input type is DECIMAL(10, 2), final Agg input
        // sum type is DECIMAL(20, 2), but output type is DECIMAL(14, 2) Spark

        // Spark use DECIMAL(20, 0) to represent long value, but we need to
        // create a SHORT_DECIMAL to get SHORT_DECIMAL result
        DCHECK(DecimalUtil::numDigits(accumulator->count) <= 18);
        auto countDecimal = UnscaledShortDecimal(accumulator->count);
        DCHECK(accumulator->sum <= LONG_MAX);
        auto longUnscaledSum = (int64_t)accumulator->sum;
        DecimalUtil::divideWithRoundUp<
            UnscaledShortDecimal,
            UnscaledShortDecimal,
            UnscaledShortDecimal>(
            average,
            UnscaledShortDecimal(longUnscaledSum),
            countDecimal,
            false,
            sumRescale,
            0);
      } else {
        VELOX_FAIL("Final Avg Agg result type must be DECIMAL");
      }
    } else if constexpr (std::is_same_v<TInputType, UnscaledShortDecimal>) {
      if constexpr (std::is_same_v<TResultType, UnscaledLongDecimal>) {
        // Spark use DECIMAL(20, 0) to represent long value
        auto countDecimal = UnscaledLongDecimal(accumulator->count);
        DCHECK(accumulator->sum <= LONG_MAX);
        auto longUnscaledSum = (int64_t)accumulator->sum;
        DecimalUtil::divideWithRoundUp<
            UnscaledLongDecimal,
            UnscaledShortDecimal,
            UnscaledLongDecimal>(
            average,
            UnscaledShortDecimal(longUnscaledSum),
            countDecimal,
            false,
            sumRescale,
            0);
      } else if constexpr (std::is_same_v<TResultType, UnscaledShortDecimal>) {
        // we enter this case when input type is DECIMAL(10, 2), final Agg input
        // sum type is DECIMAL(20, 2), but output type is DECIMAL(14, 2) Spark

        // Spark use DECIMAL(20, 0) to represent long value, but we need to
        // create a SHORT_DECIMAL to get SHORT_DECIMAL result
        DCHECK(DecimalUtil::numDigits(accumulator->count) <= 18);
        auto countDecimal = UnscaledShortDecimal(accumulator->count);
        DCHECK(accumulator->sum <= LONG_MAX);
        auto longUnscaledSum = (int64_t)accumulator->sum;
        DecimalUtil::divideWithRoundUp<
            UnscaledShortDecimal,
            UnscaledShortDecimal,
            UnscaledShortDecimal>(
            average,
            UnscaledShortDecimal(longUnscaledSum),
            countDecimal,
            false,
            sumRescale,
            0);
      } else {
        VELOX_FAIL("Final Avg Agg result type must be DECIMAL");
      }
    } else {
      VELOX_FAIL("Final Avg Agg result type must be DECIMAL");
    }

    return average;
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<TResultType>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    TResultType* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      auto accumulator = decimalAccumulator(group);
      if (isNull(group) || accumulator->count == 0) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        rawValues[i] = computeFinalValue(accumulator);
      }
    }
  }

  template <bool tableHasNulls = true>
  void mergeAccumulators(char* group, const StringView& serialized) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->mergeWith(serialized);
  }

  template <bool tableHasNulls = true, class UnscaledType>
  void mergeAccumulators(
      char* group,
      const UnscaledType& otherSum,
      const int64_t& otherCount) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    mergeSumCount(accumulator, otherSum, otherCount);
  }

  template <bool tableHasNulls = true>
  void updateNonNullValue(char* group, UnscaledLongDecimal value) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    accumulator->overflow += DecimalUtil::addWithOverflow(
        accumulator->sum, value.unscaledValue(), accumulator->sum);
    accumulator->count += 1;
  }

  template <typename UnscaledType>
  inline void mergeSumCount(
      LongDecimalWithOverflowState* accumulator,
      UnscaledType sum,
      int64_t count) {
    accumulator->count += count;
    accumulator->overflow += DecimalUtil::addWithOverflow(
        accumulator->sum, sum.unscaledValue(), accumulator->sum);
  }

  TypePtr inputType() const {
    return inputType_;
  }

 private:
  inline LongDecimalWithOverflowState* decimalAccumulator(char* group) {
    return exec::Aggregate::value<LongDecimalWithOverflowState>(group);
  }

  inline static uint8_t
  computeRescaleFactor(uint8_t fromScale, uint8_t toScale, uint8_t rScale) {
    return rScale - fromScale + toScale;
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
  const TypePtr inputType_;
};

bool registerDecimalAvgAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("ROW(DECIMAL(a_precision, a_scale), BIGINT)")
          .returnType("DECIMAL(a_precision, a_scale)")
          .build());

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_LE(
            argTypes.size(), 1, "{} takes at most one argument", name);
        auto& inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::SHORT_DECIMAL:
            if (resultType->kind() == TypeKind::SHORT_DECIMAL) {
              return std::make_unique<DecimalAverageAggregate<
                  UnscaledShortDecimal,
                  UnscaledShortDecimal>>(inputType, resultType);
            } else {
              return std::make_unique<DecimalAverageAggregate<
                  UnscaledShortDecimal,
                  UnscaledLongDecimal>>(inputType, resultType);
            }
          case TypeKind::LONG_DECIMAL:
            if (resultType->kind() == TypeKind::LONG_DECIMAL) {
              return std::make_unique<DecimalAverageAggregate<
                  UnscaledLongDecimal,
                  UnscaledLongDecimal>>(inputType, resultType);
            } else {
              VELOX_FAIL(
                  "Partial Avg Agg result type must greater than input type.");
            }
          case TypeKind::ROW: {
            DCHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            switch (sumInputType->kind()) {
              case TypeKind::LONG_DECIMAL:
                if (resultType->kind() == TypeKind::SHORT_DECIMAL) {
                  return std::make_unique<DecimalAverageAggregate<
                      UnscaledLongDecimal,
                      UnscaledShortDecimal>>(sumInputType, resultType);
                } else {
                  return std::make_unique<DecimalAverageAggregate<
                      UnscaledLongDecimal,
                      UnscaledLongDecimal>>(sumInputType, resultType);
                }
              default:
                VELOX_FAIL(
                    "Unknown sum type for {} aggregation {}",
                    name,
                    sumInputType->kindName());
            }
          }
          default:
            VELOX_FAIL(
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      });
}
} // namespace facebook::velox::functions::sparksql::aggregates
