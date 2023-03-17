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

#include "velox/exec/Aggregate.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::functions::sparksql::aggregates {

template <template <typename U, typename V, typename W> class T>
bool registerDecimalSumAggregate(const std::string& name) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .integerVariable("a_precision")
          .integerVariable("a_scale")
          .argumentType("DECIMAL(a_precision, a_scale)")
          .intermediateType("DECIMAL(a_precision, a_scale)")
          .returnType("DECIMAL(a_precision, a_scale)")
          .build(),
  };

  return exec::registerAggregateFunction(
      name,
      std::move(signatures),
      [name](
          core::AggregationNode::Step step,
          const std::vector<TypePtr>& argTypes,
          const TypePtr& resultType) -> std::unique_ptr<exec::Aggregate> {
        VELOX_CHECK_EQ(argTypes.size(), 1, "{} takes only one argument", name);
        auto inputType = argTypes[0];
        switch (inputType->kind()) {
          case TypeKind::SHORT_DECIMAL:
            return std::make_unique<
                T<UnscaledShortDecimal,
                  UnscaledLongDecimal,
                  UnscaledLongDecimal>>(resultType);
          case TypeKind::LONG_DECIMAL:
            return std::make_unique<
                T<UnscaledLongDecimal,
                  UnscaledLongDecimal,
                  UnscaledLongDecimal>>(resultType);
          case TypeKind::ROW: {
            DCHECK(!exec::isRawInput(step));
            auto sumInputType = inputType->asRow().childAt(0);
            switch (sumInputType->kind()) {
              case TypeKind::SHORT_DECIMAL:
                return std::make_unique<
                    T<UnscaledShortDecimal,
                      UnscaledLongDecimal,
                      UnscaledLongDecimal>>(resultType);
              case TypeKind::LONG_DECIMAL:
                return std::make_unique<
                    T<UnscaledLongDecimal,
                      UnscaledLongDecimal,
                      UnscaledLongDecimal>>(resultType);
              default:
                VELOX_FAIL(
                    "Unknown sum type for {} aggregation {}",
                    name,
                    sumInputType->kindName());
            }
          }
          default:
            VELOX_CHECK(
                false,
                "Unknown input type for {} aggregation {}",
                name,
                inputType->kindName());
        }
      });
}

} // namespace facebook::velox::functions::sparksql::aggregates
