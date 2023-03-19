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

template <typename TResultType, typename TInputType = TResultType>
class DecimalAggregate : public exec::Aggregate {
 public:
  explicit DecimalAggregate(TypePtr resultType) : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalAggregate);
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
          updateNonNullValue(groups[i], TResultType(value));
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
            groups[i], TResultType(decodedRaw_.valueAt<TInputType>(i)));
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      auto data = decodedRaw_.data<TInputType>();
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue<false>(groups[i], TResultType(data[i]));
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        updateNonNullValue(
            groups[i], TResultType(decodedRaw_.valueAt<TInputType>(i)));
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
          updateNonNullValue(group, TResultType(value));
        });
      } else {
        // Spark expects the result of partial avg to be non-nullable.
        exec::Aggregate::clearNull(group);
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              group, TResultType(decodedRaw_.valueAt<TInputType>(i)));
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
    auto sumCol = baseRowVector->childAt(0);
    auto countCol = baseRowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        addSingleGroupIntermediateDecimalResults(
            group,
            rows,
            sumCol->as<SimpleVector<UnscaledShortDecimal>>(),
            countCol->as<SimpleVector<int64_t>>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        addSingleGroupIntermediateDecimalResults(
            group,
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
  void addSingleGroupIntermediateDecimalResults(
      char* group,
      const SelectivityVector& rows,
      SimpleVector<UnscaledType>* sumVector,
      SimpleVector<int64_t>* countVector) {
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
    auto sumCol = rowVector->childAt(0);
    auto countCol = rowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        extractDecimalAccumulator(
            groups,
            numGroups,
            rowVector,
            sumCol->asFlatVector<UnscaledShortDecimal>(),
            countCol->asFlatVector<int64_t>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        extractDecimalAccumulator(
            groups,
            numGroups,
            rowVector,
            sumCol->asFlatVector<UnscaledLongDecimal>(),
            countCol->asFlatVector<int64_t>());
        break;
      }
      default:
        VELOX_FAIL(
            "Unsupported sum type for decimal aggregation: {}",
            sumCol->typeKind());
    }
  }

  template <class UnscaledType>
  void extractDecimalAccumulator(
      char** groups,
      int32_t numGroups,
      RowVector* rowVector,
      FlatVector<UnscaledType>* sumVector,
      FlatVector<int64_t>* countVector) {
    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    countVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(rowVector);

    int64_t* rawCounts = countVector->mutableRawValues();
    UnscaledType* rawSums = sumVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        rowVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto* accumulator = decimalAccumulator(group);
        rawCounts[i] = accumulator->count;
        rawSums[i] = (UnscaledType)accumulator->sum;
      }
    }
  }

  virtual TResultType computeFinalValue(
      LongDecimalWithOverflowState* accumulator) = 0;

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
  void updateNonNullValue(char* group, TResultType value) {
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

 private:
  inline LongDecimalWithOverflowState* decimalAccumulator(char* group) {
    return exec::Aggregate::value<LongDecimalWithOverflowState>(group);
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
};

template <typename TUnscaledType>
class DecimalAverageAggregate : public DecimalAggregate<TUnscaledType> {
 public:
  explicit DecimalAverageAggregate(TypePtr resultType)
      : DecimalAggregate<TUnscaledType>(resultType) {}

  virtual TUnscaledType computeFinalValue(
      LongDecimalWithOverflowState* accumulator) final {
    // Handles round-up of fraction results.
    int128_t average{0};
    DecimalUtil::computeAverage(
        average, accumulator->sum, accumulator->count, accumulator->overflow);
    return TUnscaledType(average);
  }
};

bool registerDecimalAvgAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures;
  signatures.push_back(exec::AggregateFunctionSignatureBuilder()
                           .integerVariable("a_precision")
                           .integerVariable("a_scale")
                           .argumentType("DECIMAL(a_precision, a_scale)")
                           .intermediateType("VARBINARY")
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
            return std::make_unique<
                DecimalAverageAggregate<UnscaledShortDecimal>>(resultType);
          case TypeKind::LONG_DECIMAL:
            return std::make_unique<
                DecimalAverageAggregate<UnscaledLongDecimal>>(resultType);
          case TypeKind::ROW: {
            DCHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            switch (sumInputType->kind()) {
              case TypeKind::SHORT_DECIMAL:
                return std::make_unique<
                    DecimalAverageAggregate<UnscaledShortDecimal>>(resultType);
              case TypeKind::LONG_DECIMAL:
                return std::make_unique<
                    DecimalAverageAggregate<UnscaledLongDecimal>>(resultType);
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
