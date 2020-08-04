// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "transformations/low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class ConcatFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2);

    static std::shared_ptr<ngraph::Function> getOriginalWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3);

    static std::shared_ptr<ngraph::Function> getOriginalWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getOriginalSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparent,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantize1,
        const FakeQuantizeOnData& fakeQuantize2,
        const DequantizationOperations& dequantizationOperations);

    static std::shared_ptr<ngraph::Function> getReferenceWithNeighbors(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparentIntermediate,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

    static std::shared_ptr<ngraph::Function> getReferenceSelectionWithIntermediate(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const bool transparent,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations& dequantizationBeforePooling,
        const DequantizationOperations& dequantizationAfterPooling);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
