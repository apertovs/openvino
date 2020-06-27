// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/fake_quantize_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/multiply_add.hpp"
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

using namespace ngraph::pass;

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
        fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputHighValues);
    fakeQuantize->set_friendly_name("fakeQuantize");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fakeQuantize) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

std::shared_ptr<ngraph::Function> FakeQuantizeFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const FakeQuantizeOnData& fakeQuantizeOnData,
    const std::vector<float>& expectedSubtractValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    input->set_friendly_name("input");

    // TODO: way to instantiate TypeRelaxed FakeQuantize
    // TODO: use wrapper later
    auto inputLowNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.inputLowValues, fakeQuantizeOnData.inputLowValues.empty());
    auto inputHighNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.inputHighValues, fakeQuantizeOnData.inputHighValues.empty());
    auto outputLowNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.outputLowValues, fakeQuantizeOnData.outputLowValues.empty());
    auto outputHighNode = ngraph::builder::makeConstant(
        precision, fakeQuantizeOnData.constantShape, fakeQuantizeOnData.outputHighValues, fakeQuantizeOnData.outputHighValues.empty());
    // auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
    //    input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fakeQuantizeOnData.quantizationLevel);
    std::shared_ptr<ngraph::opset1::FakeQuantize> fakeQuantize = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::FakeQuantize>>(
        input, inputLowNode, inputHighNode, outputLowNode, outputHighNode, fakeQuantizeOnData.quantizationLevel);

    // auto fakeQuantize = ngraph::builder::makeFakeQuantize(
    //    input, precision, fakeQuantizeOnData.quantizationLevel, fakeQuantizeOnData.constantShape,
    //    fakeQuantizeOnData.lowValues, fakeQuantizeOnData.highValues, fakeQuantizeOnData.lowValues, fakeQuantizeOnData.highValues);


    // auto quantizeConvert = ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(fakeQuantize, expectedPrecision);
    // auto quantizeConvert1 = low_precision::fold<ngraph::opset1::Convert>(fakeQuantize, params.precisionsOnActivations[0]);
    auto quantizeConvert2 = std::make_shared<ngraph::opset1::Convert>(fakeQuantize, precision);

    // why all children change precision?
    // auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(fakeQuantize);
    // relaxed_layer->set_overriden_output_type(expectedPrecision);
    // std::dynamic_pointer_cast<ngraph::Node>(fakeQuantize)->validate_and_infer_types();

    // fakeQuantize = ngraph::pass::low_precision::fold<ngraph::opset1::Convert>(fakeQuantize, precision);

    // copy_runtime_info(layer, replacement);
    // replace_node(layer, replacement);

    // fakeQuantize->set_friendly_name("fakeQuantize");
    // fakeQuantize->set_output_type(0, expectedPrecision, inputShape);
    // setOutDataPrecision(fakeQuantize, expectedPrecision);

    // const auto convert = std::make_shared<ngraph::opset1::Convert>(fakeQuantize, precision);

    // TODO: MultiplyAdd constant shape is hardcoded
    // auto subtract = std::make_shared<ngraph::op::TypeRelaxed<ngraph::op::Subtract>>(
    std::shared_ptr<ngraph::opset1::Subtract> subtract = expectedSubtractValues.empty() ?
        nullptr :
        std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Subtract>>(
            quantizeConvert2, // fakeQuantize
            ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 0.f }),
            ngraph::op::AutoBroadcastSpec::NUMPY);

    std::shared_ptr<ngraph::opset1::Multiply> multiply;
    if (subtract != nullptr) {
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(subtract, precision);

        multiply = std::make_shared<ngraph::opset1::Multiply>(
            subtract,
            ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 1.f }));
    } else {
        multiply = std::make_shared<ngraph::opset1::Multiply>(
            quantizeConvert2,
            ngraph::opset1::Constant::create(precision, ngraph::Shape{ }, { 1.f }));
    }

    // TODO: just to debug
    auto outputs = fakeQuantize->get_output_partial_shape(0);

    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(fakeQuantize, params.precisionsOnActivations[0]);
    // ngraph::pass::low_precision::NetworkHelper::removeLayer(quantizeConvert1);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "FakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
