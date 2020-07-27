// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "common/fake_quantize_on_data.hpp"
#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class SqueezeFunction {
public:
    class LayerDescription {
    public:
        std::vector<float> values;
        ngraph::Shape shape;
    };
    class ActualValues {
    public:
        ngraph::element::Type lowPrecision;
        LayerDescription subtract;
        LayerDescription mutliply;
    };

    class ExpectedValues {
    public:
        ngraph::element::Type activationPrecision;
        LayerDescription subtract;
        LayerDescription mutliply;
    };

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const std::vector<size_t>& axes,
        const ActualValues& values);

    static std::shared_ptr<ngraph::Function> SqueezeFunction::getOriginal(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const FakeQuantizeOnData& fakeQuantizeOnData,
        const std::vector<size_t>& axes);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type originalFunctionPrecision,
        const ngraph::Shape& inputShape,
        const std::vector<size_t>& axes,
        const ExpectedValues& values);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
