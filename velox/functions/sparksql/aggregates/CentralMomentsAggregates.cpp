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

#include "velox/exec/Aggregate.h"
#include "velox/functions/lib/aggregates/CentralMomentsAggregate.h"

using namespace facebook::velox::functions::aggregate;

namespace facebook::velox::functions::aggregate::sparksql {

struct SkewnessResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    return accumulator.count() >= 1 && accumulator.m2() != 0;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    return std::sqrt(accumulator.count()) * accumulator.m3() /
        std::pow(accumulator.m2(), 1.5);
  }
};

struct KurtosisResultAccessor {
  static bool hasResult(const CentralMomentsAccumulator& accumulator) {
    return accumulator.count() >= 1 && accumulator.m2() != 0;
  }

  static double result(const CentralMomentsAccumulator& accumulator) {
    double count = accumulator.count();
    double m2 = accumulator.m2();
    double m4 = accumulator.m4();
    return ((count - 1) * count * (count + 1)) / ((count - 2) * (count - 3)) *
        m4 / (m2 * m2) -
        3 * ((count - 1) * (count - 1)) / ((count - 2) * (count - 3));
  }
};

void registerCentralMomentsAggregates(const std::string& prefix) {
  registerCentralMoments<KurtosisResultAccessor>(prefix + "kurtosis");
  registerCentralMoments<SkewnessResultAccessor>(prefix + "skewness");
}

} // namespace facebook::velox::aggregate::sparksql
