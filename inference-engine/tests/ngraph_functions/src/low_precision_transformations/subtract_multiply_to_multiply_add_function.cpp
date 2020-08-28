// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/subtract_multiply_to_multiply_add_function.hpp"

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"
#include "low_precision_transformations/network_helper.hpp"

using ngraph::pass::low_precision::make_dequantization_operation;

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    dequantizationOp->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(dequantizationOp) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getOriginal(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precision,
    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        ngraph::Shape(inputShape));

    const std::shared_ptr<Node> fq = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape1 = std::make_shared<ngraph::opset1::Reshape>(
        fq,
        std::make_shared<ngraph::opset1::Constant>(
            ngraph::element::i64,
            Shape({ 3 }),
            std::vector<int64_t>({ static_cast<int64_t>(inputShape[0]), static_cast<int64_t>(inputShape[1]), -1 })),
        false);

    const std::shared_ptr<ngraph::opset1::Reshape> reshape2 = std::make_shared<ngraph::opset1::Reshape>(
        reshape1,
        std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, Shape({ 4 }), inputShape),
        false);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(reshape2) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

std::shared_ptr<ngraph::Function> SubtractMultiplyToMultiplyAddFunction::getReference(
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const ngraph::builder::subgraph::DequantizationOperations& dequantization,
    const ngraph::element::Type precisionAfterDequantization,
    const ngraph::builder::subgraph::Multiply& multiply,
    const ngraph::builder::subgraph::Add& add) {
    const std::shared_ptr<op::v0::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        ngraph::Shape(inputShape));

    std::shared_ptr<Node> dequantizationOp = makeDequantization(input, dequantization);
    std::shared_ptr<Node> parent = dequantizationOp;

    if (!multiply.empty()) {
        auto constant = std::make_shared<ngraph::opset1::Constant>(multiply.outPrecision, multiply.constantShape, multiply.values);
        parent = std::make_shared<op::TypeRelaxed<ngraph::opset1::Multiply>>(
            std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get());
    }

    if (!add.empty()) {
        auto constant = std::make_shared<ngraph::opset1::Constant>(add.outPrecision, add.constantShape, add.values);

        parent = std::make_shared<op::TypeRelaxed<ngraph::opset1::Add>>(
            std::vector<element::Type>{element::f32, element::f32}, std::vector<element::Type>{},
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(constant, element::f32).get());
    }
    parent->set_friendly_name("output");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(parent) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "SubtractMultiplyToMultiplyAddFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
