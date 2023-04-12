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
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

using namespace facebook::velox;
using namespace facebook::velox::exec::test;

class WindowTopKFilterTest : public OperatorTestBase {};

TEST_F(WindowTopKFilterTest, singleKeyTop3Aesc) {
  int32_t partitionSize = 100;
  int32_t partitionNum = 10;
  int32_t topK = 3;

  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < partitionNum; ++i) {
    std::vector<int32_t> c0(partitionSize);
    // c0 = {i+1, i+1, i+1, ..., i+1}
    c0.assign(partitionSize, i + 1);
    std::vector<int32_t> c1(partitionSize);
    for (int32_t j = 0; j < partitionSize; ++j) {
      c1[j] = j + 1;
    }
    std::random_device rd;
    std::shuffle(c1.begin(), c1.end(), std::default_random_engine{rd()});
    vectors.push_back(makeRowVector({makeFlatVector<int32_t>(c0), makeFlatVector<int32_t>(c1)}));
  }

  auto sql = fmt::format("c1 {}", "NULLS LAST");
  auto plan = PlanBuilder().values(vectors).windowTopKFilter(topK, {"c0"}, {sql}).planNode();

  std::vector<int32_t> ec0;
  std::vector<int32_t> ec1;
  for (int32_t i = 0; i < partitionNum; ++i) {
    std::vector<int32_t> c0(topK);
    std::vector<int32_t> c1 = {1, 2, 3};
    c0.assign(topK, i + 1);
    ec0.insert(ec0.end(), c0.begin(), c0.end());
    ec1.insert(ec1.end(), c1.begin(), c1.end());
  }
  auto expect = makeRowVector({makeFlatVector<int32_t>(ec0), makeFlatVector<int32_t>(ec1)});
  assertQuery(plan, expect);
}

TEST_F(WindowTopKFilterTest, singleKeyTop5Desc) {
  int32_t partitionSize = 100;
  int32_t partitionNum = 10;
  int32_t topK = 5;

  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < partitionNum; ++i) {
    std::vector<int32_t> c0(partitionSize);
    // c0 = {i+1, i+1, i+1, ..., i+1}
    c0.assign(partitionSize, i + 1);
    std::vector<int32_t> c1(partitionSize);
    for (int32_t j = 0; j < partitionSize; ++j) {
      c1[j] = j + 1;
    }
    std::random_device rd;
    std::shuffle(c1.begin(), c1.end(), std::default_random_engine{rd()});
    vectors.push_back(makeRowVector({makeFlatVector<int32_t>(c0), makeFlatVector<int32_t>(c1)}));
  }

  auto sql = fmt::format("c1 {}", "DESC NULLS LAST");
  auto plan = PlanBuilder().values(vectors).windowTopKFilter(topK, {"c0"}, {sql}).planNode();

  std::vector<int32_t> ec0;
  std::vector<int32_t> ec1;
  for (int32_t i = 0; i < partitionNum; ++i) {
    std::vector<int32_t> c0(topK);
    std::vector<int32_t> c1 = {100, 99, 98, 97, 96};
    c0.assign(topK, i + 1);
    ec0.insert(ec0.end(), c0.begin(), c0.end());
    ec1.insert(ec1.end(), c1.begin(), c1.end());
  }
  auto expect = makeRowVector({makeFlatVector<int32_t>(ec0), makeFlatVector<int32_t>(ec1)});
  assertQuery(plan, expect);
}

TEST_F(WindowTopKFilterTest, multipleKeyTop3Aesc) {
  int32_t partitionSize = 30;
  int32_t partitionNum = 5;
  int32_t secondDistinctKeyNum = 3;
  int32_t topK = 3;

  std::vector<RowVectorPtr> vectors;
  for (int32_t i = 0; i < partitionNum; ++i) {
    std::vector<int32_t> c0(partitionSize);
    // c0 = {i+1, i+1, i+1, ..., i+1}
    c0.assign(partitionSize, i + 1);

    // c1 = {1, 1, ...,2, 2, ..., 3, 3, ... }
    std::vector<int64_t> c1(partitionSize);
    // c2 = {random value from 1 to 10, random value from 1 to 10, ...}
    std::vector<int32_t> c2(partitionSize);

    for (int32_t j = 0; j < secondDistinctKeyNum; ++j) {
      for (int32_t k = 0; k < (partitionSize / secondDistinctKeyNum); ++k) {
        c1[j * (partitionSize / secondDistinctKeyNum) + k] = j + 1;
      }
      for (int32_t k = 0; k < (partitionSize / secondDistinctKeyNum); ++k) {
        c2[j * (partitionSize / secondDistinctKeyNum) + k] = k + 1;
      }
      auto begin = c2.begin() + j * (partitionSize / secondDistinctKeyNum);
      auto end = c2.begin() + (j + 1) * (partitionSize / secondDistinctKeyNum);
      std::random_device rd;
      std::shuffle(begin, end, std::default_random_engine{rd()});
    }
    vectors.push_back(makeRowVector({makeFlatVector<int32_t>(c0), makeFlatVector<int64_t>(c1), makeFlatVector<int32_t>(c2)}));
  }

  auto sql = fmt::format("c2 {}", "NULLS LAST");
  auto plan = PlanBuilder().values(vectors).windowTopKFilter(topK, {"c0", "c1"}, {sql}).planNode();

  std::vector<int32_t> ec0;
  std::vector<int64_t> ec1;
  std::vector<int32_t> ec2;
  for (int32_t i = 0; i < partitionNum; ++i) {
    for (int32_t j = 0; j < secondDistinctKeyNum; ++j) {
      std::vector<int32_t> c0(topK);
      std::vector<int64_t> c1(topK);
      std::vector<int32_t> c2 = {1, 2, 3};
      c0.assign(topK, i + 1);
      c1.assign(topK, j + 1);
      ec0.insert(ec0.end(), c0.begin(), c0.end());
      ec1.insert(ec1.end(), c1.begin(), c1.end());
      ec2.insert(ec2.end(), c2.begin(), c2.end());
    }
  }

  auto expect = makeRowVector({makeFlatVector<int32_t>(ec0), makeFlatVector<int64_t>(ec1), makeFlatVector<int32_t>(ec2)});
  assertQuery(plan, expect);
}