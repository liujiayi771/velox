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
#include <iostream>
#include "velox/exec/WindowTopKFilter.h"

namespace facebook::velox::exec {

WindowTopKFilter::WindowTopKFilter(
    int32_t operatorId,
    DriverCtx* driverCtx,
    const std::shared_ptr<const core::WindowTopNFilterNode>& topKNode)
    : Operator(
          driverCtx,
          topKNode->outputType(),
          operatorId,
          topKNode->id(),
          "WindowTopKFilter"),
      k_(topKNode->k()),
      data_(std::make_unique<RowContainer>(outputType_->children(), pool())),
      partitionKeyComparator_(
          outputType_,
          topKNode->partitionKeys(),
          {},
          data_.get()),
      sortKeyComparator_(
          outputType_,
          topKNode->sortingKeys(),
          topKNode->sortingOrders(),
          data_.get()),
      eachKeyTopRows_(partitionKeyComparator_),
      decodedVectors_(outputType_->children().size()) {}

WindowTopKFilter::Comparator::Comparator(
    const RowTypePtr& type,
    const std::vector<std::shared_ptr<const core::FieldAccessTypedExpr>>&
        sortingKeys,
    const std::vector<core::SortOrder>& sortingOrders,
    RowContainer* rowContainer)
    : rowContainer_(rowContainer) {
  const core::SortOrder defaultPartitionSortOrder(true, true);
  auto numKeys = sortingKeys.size();
  for (int i = 0; i < numKeys; ++i) {
    auto channel = exprToChannel(sortingKeys[i].get(), type);
    VELOX_CHECK(
        channel != kConstantChannel,
        "WindowTopKFilter doesn't allow constant comparison keys");
    if (i < sortingOrders.size()) {
      keyInfo_.push_back(std::make_pair(channel, sortingOrders[i]));
    } else {
      keyInfo_.push_back(std::make_pair(channel, defaultPartitionSortOrder));
    }
  }
}

void WindowTopKFilter::addInput(RowVectorPtr input) {
  SelectivityVector allRows(input->size());

  for (int col = 0; col < input->childrenSize(); ++col) {
    decodedVectors_[col].decode(*input->childAt(col), allRows);
  }

  char* currentRow = data_->newRow();
  for (int row = 0; row < input->size(); ++row) {
    for (int col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedVectors_[col], row, currentRow, col);
    }

    auto topRows = eachKeyTopRows_.find(currentRow);
    if (topRows != eachKeyTopRows_.end()) {
      char* topRow = topRows->second.top();
      if (topRows->second.size() >= k_) {
        if (!sortKeyComparator_(decodedVectors_, row, topRow)) {
          data_->initializeRow(currentRow, true);
          continue;
        }
        topRows->second.pop();
        topRows->second.push(currentRow);
        if (topRow == topRows->first) {
          // we can not reuse the top row memory if it is the map key, because
          // it will be used to identify the partition
          currentRow = data_->newRow();
        } else {
          // reuse topRow to store next input row
          currentRow = data_->initializeRow(topRow, true);
        }
      } else {
        topRows->second.push(currentRow);
        totalOutputSize_ += 1;
        currentRow = data_->newRow();
      }
    } else {
      auto queue = std::priority_queue<char*, std::vector<char*>, Comparator>(sortKeyComparator_);
      queue.push(currentRow);
      totalOutputSize_ += 1;
      eachKeyTopRows_.emplace(currentRow, queue);
      currentRow = data_->newRow();
    }
  }

  int total = 0;
  for (auto [k, v] : eachKeyTopRows_) {
    total += v.size();
  }
  std::cout << "total=" << total << std::endl;
}

RowVectorPtr WindowTopKFilter::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  uint32_t numRowsToReturn =
      std::min(kMaxNumRowsToReturn, rows_.size() - numRowsReturned_);
  VELOX_CHECK(numRowsToReturn > 0);

  auto result = std::dynamic_pointer_cast<RowVector>(
      BaseVector::create(outputType_, numRowsToReturn, operatorCtx_->pool()));

  for (int i = 0; i < outputType_->size(); ++i) {
    data_->extractColumn(
        rows_.data() + numRowsReturned_,
        numRowsToReturn,
        i,
        result->childAt(i));
  }
  numRowsReturned_ += numRowsToReturn;
  finished_ = (numRowsReturned_ == rows_.size());
  return result;
}

void WindowTopKFilter::noMoreInput() {
  Operator::noMoreInput();
  if (eachKeyTopRows_.empty()) {
    finished_ = true;
    return;
  }
  std::cout << "totalSize=" << totalOutputSize_ << std::endl;
  rows_.resize(totalOutputSize_);
  int row = 0;
  for (auto [key, val] : eachKeyTopRows_) {
    while (!val.empty()) {
      rows_[row++] = val.top();
      val.pop();
    }
  }
}

bool WindowTopKFilter::isFinished() {
  return finished_;
}
} // namespace facebook::velox::exec
