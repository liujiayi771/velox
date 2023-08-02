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
#include "velox/common/base/tests/GTestUtils.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/functions/lib/aggregates/tests/AggregationTestBase.h"
#include "velox/functions/sparksql/aggregates/Register.h"

using namespace facebook::velox::exec::test;
using namespace facebook::velox::functions::aggregate::test;

namespace facebook::velox::functions::aggregate::sparksql::test {

class CentralMomentsAggregationTest
    : public virtual AggregationTestBase,
      public testing::WithParamInterface<std::string> {
 protected:
  void SetUp() override {
    AggregationTestBase::SetUp();
    allowInputShuffle();
    registerAggregateFunctions("spark_");
  }

  void testSingleColGlobalAgg(
      const std::string& aggName,
      const RowVectorPtr& data,
      const std::vector<RowVectorPtr>& expectedResult) {
    auto partialAgg = fmt::format("spark_{0}(c0)", aggName);
    testAggregations({data}, {}, {partialAgg}, expectedResult);
  }
};

TEST_F(CentralMomentsAggregationTest, skewnessHasResult) {
  // In Spark, count equals to 2 will not get NULL output.
  auto data =
      makeRowVector({makeFlatVector<double>({1.0f, 0.0f, 0.0f, 3.0f, 0.0f})});
  for (auto i = 0; i < 5; i++) {
    if (data->childAt(0)->asFlatVector<double>()->valueAt(i) == 0.0f) {
      data->childAt(0)->setNull(i, true);
    }
  }

  auto expectedResult = makeRowVector({makeFlatVector<double>(0)});
  testSingleColGlobalAgg("skewness", data, {expectedResult});
}

TEST_F(CentralMomentsAggregationTest, kurtosisHasResult) {
  // In Spark, count equals to 2 will not get NULL output.
  auto data =
      makeRowVector({makeFlatVector<double>({1.0f, 0.0f, 0.0f, 3.0f, 0.0f})});
  for (auto i = 0; i < 5; i++) {
    if (data->childAt(0)->asFlatVector<double>()->valueAt(i) == 0.0f) {
      data->childAt(0)->setNull(i, true);
    }
  }

  auto expectedResult = makeRowVector({makeFlatVector<double>(-2)});
  testSingleColGlobalAgg("kurtosis", data, {expectedResult});
}

TEST_F(CentralMomentsAggregationTest, constantInput) {
  // When the input is a constant, m2 will equal to 0.
  vector_size_t size = 10;
  auto data = makeRowVector(
      {makeFlatVector<int32_t>(size, [](auto row) { return 5; })});
  auto expectedResult = makeRowVector({makeAllNullFlatVector<double>(1)});
  testSingleColGlobalAgg("skewness", data, {expectedResult});
  testSingleColGlobalAgg("kurtosis", data, {expectedResult});
}
} // namespace facebook::velox::aggregate::test
