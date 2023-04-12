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

#include "velox/exec/Operator.h"
#include "velox/exec/RowContainer.h"

namespace facebook::velox::exec {

class WindowTopKFilter : public Operator {
 public:
    WindowTopKFilter(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::WindowTopNFilterNode>& topKNode);

  bool needsInput() const override {
    return !noMoreInput_;
  }

  void addInput(RowVectorPtr input) override;

  RowVectorPtr getOutput() override;

  void noMoreInput() override;

  BlockingReason isBlocked(ContinueFuture* /*future*/) override {
    return BlockingReason::kNotBlocked;
  }

  bool isFinished() override;

 private:
  static constexpr size_t kMaxNumRowsToReturn = 1024;
  class Comparator {
   public:
    Comparator(
        const RowTypePtr& outputType,
        const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
            sortingKeys,
        const std::vector<core::SortOrder>& sortingOrders,
        RowContainer* rowContainer);

    // Returns true if lhs < rhs, false otherwise.
    bool operator()(const char* lhs, const char* rhs) const {
      if (lhs == rhs) {
        return false;
      }
      for (auto& key : keyInfo_) {
        if (auto result = rowContainer_->compare(
                lhs,
                rhs,
                key.first,
                {key.second.isNullsFirst(), key.second.isAscending(), false})) {
          return result < 0;
        }
      }
      return false;
    }

    // Returns true if decodeVectors[index] < rhs, false otherwise.
    bool operator()(
        const std::vector<DecodedVector>& decodedVectors,
        vector_size_t index,
        const char* rhs) const {
      for (auto& key : keyInfo_) {
        if (auto result = rowContainer_->compare(
                rhs,
                rowContainer_->columnAt(key.first),
                decodedVectors[key.first],
                index,
                {key.second.isNullsFirst(), key.second.isAscending(), false})) {
          return result > 0;
        }
      }
      return false;
    }

   private:
    std::vector<std::pair<column_index_t, core::SortOrder>> keyInfo_;
    RowContainer* rowContainer_;
  };

  std::vector<std::pair<column_index_t, core::SortOrder>> partitionKeyInfo_;
  const int32_t k_;

  bool finished_ = false;
  uint32_t numRowsReturned_ = 0;
  uint32_t maxCardinality_ = 100;

  uint64_t totalOutputSize_ = 0;

  // we use this RowContainer to store input row temporarily
  std::unique_ptr<RowContainer> data_;
  Comparator partitionKeyComparator_;
  Comparator sortKeyComparator_;
  std::map<char*, std::priority_queue<char*, std::vector<char*>, Comparator>, Comparator> eachKeyTopRows_;

  std::vector<char*> rows_;

  std::vector<DecodedVector> decodedVectors_;
};
} // namespace facebook::velox::exec
