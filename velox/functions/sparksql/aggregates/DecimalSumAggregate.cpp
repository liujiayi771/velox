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

#include "velox/common/base/IOUtils.h"
#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::functions::sparksql::aggregates {

/**
 *  LongDecimalWithOverflowState has the following fields:
 *    SUM: Total sum so far.
 *    COUNT: Total number of rows so far.
 *    OVERFLOW: Total count of net overflow or underflow so far.
 */
struct LongDecimalWithOverflowState {
 public:
  void mergeWith(const StringView& serializedData) {
    VELOX_CHECK_EQ(serializedData.size(), serializedSize());
    auto serialized = serializedData.data();
    common::InputByteStream stream(serialized);
    count += stream.read<int64_t>();
    overflow += stream.read<int64_t>();
    uint64_t lowerSum = stream.read<uint64_t>();
    int64_t upperSum = stream.read<int64_t>();
    int128_t result = 0;
    overflow += DecimalUtil::addWithOverflow(
        result, buildInt128(upperSum, lowerSum), this->sum());
    this->setSum(result);
  }

  template <class UnscaledType>
  void mergeWith(const UnscaledType otherSum, int64_t otherCount) {
    count += otherCount;
    int128_t result = 0;
    int128_t sum = otherSum.unscaledValue();
    overflow += DecimalUtil::addWithOverflow(result, sum, this->sum());
    this->setSum(result);
  }

  void serialize(StringView& serialized) {
    VELOX_CHECK_EQ(serialized.size(), serializedSize());
    char* outputBuffer = const_cast<char*>(serialized.data());
    common::OutputByteStream outStream(outputBuffer);
    outStream.append((char*)&count, sizeof(int64_t));
    outStream.append((char*)&overflow, sizeof(int64_t));
    outStream.append((char*)&lowerSum, sizeof(int64_t));
    outStream.append((char*)&upperSum, sizeof(int64_t));
  }

  /*
   * Total size = sizeOf(count) + sizeOf(overflow) +
   *              sizeOf(upperSum) + sizeOf(lowerSum)
   *            = 8 + 8 + 8 + 8 = 32.
   */
  inline static size_t serializedSize() {
    return sizeof(int64_t) * 4;
  }

  inline int128_t sum() {
    return buildInt128(upperSum, lowerSum);
  }

  inline void setSum(int128_t& sum) {
    upperSum = UPPER(sum);
    lowerSum = LOWER(sum);
  }

  // The accumulator's sum is int128_t; however, it is maintained as two int64_t
  // values because int128_t causes the issue with alignment resulting in
  // segfault.
  // Certain operations such as placement new and reinterpret_cast used in the
  // aggregation framework require int128_t to be aligned. It is not trivial to
  // achieve this alignment given the layout of the aggregates in RowContainer.
  int64_t upperSum{0};
  int64_t lowerSum{0};
  int64_t count{0};
  int64_t overflow{0};
};

template <typename TInputType>
class DecimalSumAggregate : public exec::Aggregate {
 public:
  explicit DecimalSumAggregate(TypePtr resultType) : exec::Aggregate(resultType) {}

  int32_t accumulatorFixedWidthSize() const override {
    return sizeof(DecimalSumAggregate);
  }

  void initializeNewGroups(
      char** groups,
      folly::Range<const vector_size_t*> indices) override {
    setAllNulls(groups, indices);
    for (auto i : indices) {
      new (groups[i] + offset_) LongDecimalWithOverflowState();
    }
  }

  virtual UnscaledLongDecimal computeFinalValue(
      LongDecimalWithOverflowState* accumulator) final {
    // Value is valid if the conditions below are true.
    int128_t sum = accumulator->sum();
    if ((accumulator->overflow == 1 && sum < 0) ||
        (accumulator->overflow == -1 && sum > 0)) {
      sum = static_cast<int128_t>(
          DecimalUtil::kOverflowMultiplier * accumulator->overflow + sum);
      accumulator->setSum(sum);
    } else {
      VELOX_CHECK(accumulator->overflow == 0, "Decimal overflow");
    }

    VELOX_CHECK(UnscaledLongDecimal::valueInRange(sum), "Decimal overflow");
    return UnscaledLongDecimal(sum);
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
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
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
      }
    } else if (decodedRaw_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (!decodedRaw_.isNullAt(i)) {
          updateNonNullValue(
              group, UnscaledLongDecimal(decodedRaw_.valueAt<TInputType>(i)));
        }
      });
    } else if (!exec::Aggregate::numNulls_ && decodedRaw_.isIdentityMapping()) {
      const TInputType* data = decodedRaw_.data<TInputType>();
      LongDecimalWithOverflowState accumulator;
      rows.applyToSelected([&](vector_size_t i) {
        int128_t result = 0;
        accumulator.overflow += DecimalUtil::addWithOverflow(
            result, data[i].unscaledValue(), accumulator.sum());
        accumulator.setSum(result);
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
        int128_t result = 0;
        auto input = decodedRaw_.valueAt<TInputType>(i).unscaledValue();
        accumulator.overflow += DecimalUtil::addWithOverflow(
            result,
            input,
            accumulator.sum());
        accumulator.setSum(result);
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
    auto isEmptyCol = baseRowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        addIntermediateDecimalResults(
            groups,
            rows,
            sumCol->as<SimpleVector<UnscaledShortDecimal>>(),
            isEmptyCol->as<SimpleVector<bool>>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        addIntermediateDecimalResults(
            groups,
            rows,
            sumCol->as<SimpleVector<UnscaledLongDecimal>>(),
            isEmptyCol->as<SimpleVector<bool>>());
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
      SimpleVector<bool>* isEmptyVector) {
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          auto accumulator = decimalAccumulator(groups[i]);
          accumulator->mergeWith(sum, 0);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        accumulator->mergeWith(sum, 0);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(groups[i]);
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        auto accumulator = decimalAccumulator(groups[i]);
        accumulator->mergeWith(sum, 0);
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
    auto isEmptyCol = baseRowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        addSingleGroupIntermediateDecimalResults(
            group,
            rows,
            sumCol->as<SimpleVector<UnscaledShortDecimal>>(),
            isEmptyCol->as<SimpleVector<bool>>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        addSingleGroupIntermediateDecimalResults(
            group,
            rows,
            sumCol->as<SimpleVector<UnscaledLongDecimal>>(),
            isEmptyCol->as<SimpleVector<bool>>());
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
      SimpleVector<bool>* isEmptyVector) {
    if (decodedPartial_.isConstantMapping()) {
      if (!decodedPartial_.isNullAt(0)) {
        auto decodedIndex = decodedPartial_.index(0);
        auto sum = sumVector->valueAt(decodedIndex);
        rows.applyToSelected([&](vector_size_t i) {
          mergeAccumulators(group, sum, 0);
        });
      }
    } else if (decodedPartial_.mayHaveNulls()) {
      rows.applyToSelected([&](vector_size_t i) {
        if (decodedPartial_.isNullAt(i)) {
          return;
        }
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, 0);
      });
    } else {
      rows.applyToSelected([&](vector_size_t i) {
        clearNull(group);
        auto decodedIndex = decodedPartial_.index(i);
        auto sum = sumVector->valueAt(decodedIndex);
        mergeAccumulators(group, sum, 0);
      });
    }
  }

  void extractAccumulators(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto rowVector = (*result)->as<RowVector>();
    auto sumCol = rowVector->childAt(0);
    auto isEmptyCol = rowVector->childAt(1);
    switch (sumCol->typeKind()) {
      case TypeKind::SHORT_DECIMAL: {
        extractDecimalAccumulator(
            groups,
            numGroups,
            rowVector,
            sumCol->asFlatVector<UnscaledShortDecimal>(),
            isEmptyCol->asFlatVector<bool>());
        break;
      }
      case TypeKind::LONG_DECIMAL: {
        extractDecimalAccumulator(
            groups,
            numGroups,
            rowVector,
            sumCol->asFlatVector<UnscaledLongDecimal>(),
            isEmptyCol->asFlatVector<bool>());
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
      FlatVector<bool>* isEmptyVector) {
    rowVector->resize(numGroups);
    sumVector->resize(numGroups);
    isEmptyVector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(sumVector);

    UnscaledType* rawSums = sumVector->mutableRawValues();
    bool* rawIsEmpty = isEmptyVector->mutableRawValues();
    for (auto i = 0; i < numGroups; ++i) {
      if (isNull(groups[i])) {
        sumVector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto accumulator = decimalAccumulator(groups[i]);
        rawSums[i] = (UnscaledType)accumulator->sum();
        rawIsEmpty[i] = false;
      }
    }
  }

  void extractValues(char** groups, int32_t numGroups, VectorPtr* result)
      override {
    auto vector = (*result)->as<FlatVector<UnscaledLongDecimal>>();
    VELOX_CHECK(vector);
    vector->resize(numGroups);
    uint64_t* rawNulls = getRawNulls(vector);

    UnscaledLongDecimal* rawValues = vector->mutableRawValues();
    for (int32_t i = 0; i < numGroups; ++i) {
      char* group = groups[i];
      if (isNull(group)) {
        vector->setNull(i, true);
      } else {
        clearNull(rawNulls, i);
        auto accumulator = decimalAccumulator(group);
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
    accumulator->mergeWith(otherSum, otherCount);
  }

  template <bool tableHasNulls = true>
  void updateNonNullValue(char* group, UnscaledLongDecimal value) {
    if constexpr (tableHasNulls) {
      exec::Aggregate::clearNull(group);
    }
    auto accumulator = decimalAccumulator(group);
    int128_t result = 0;
    accumulator->overflow += DecimalUtil::addWithOverflow(
        result, value.unscaledValue(), accumulator->sum());
    accumulator->setSum(result);
    accumulator->count += 1;
  }

 private:
  inline LongDecimalWithOverflowState* decimalAccumulator(char* group) {
    return exec::Aggregate::value<LongDecimalWithOverflowState>(group);
  }

  DecodedVector decodedRaw_;
  DecodedVector decodedPartial_;
};

bool registerDecimalSumAggregate(const std::string& name) {
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
                DecimalSumAggregate<UnscaledShortDecimal>>(resultType);
          case TypeKind::LONG_DECIMAL:
            return std::make_unique<
                DecimalSumAggregate<UnscaledLongDecimal>>(resultType);
          case TypeKind::ROW: {
            DCHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            switch (sumInputType->kind()) {
              case TypeKind::SHORT_DECIMAL:
                return std::make_unique<
                    DecimalSumAggregate<UnscaledShortDecimal>>(resultType);
              case TypeKind::LONG_DECIMAL:
                return std::make_unique<
                    DecimalSumAggregate<UnscaledLongDecimal>>(resultType);
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
} // namespace facebook::velox::aggregate
