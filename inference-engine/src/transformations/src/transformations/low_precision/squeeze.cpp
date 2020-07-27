// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/low_precision/squeeze.hpp"

#include <memory>
#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

#include "transformations/low_precision/network_helper.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

SqueezeTransformation::SqueezeTransformation(const Params& params) : LayerTransformation(params) {
}

void SqueezeTransformation::registerMatcherIn(GraphRewrite &pass, TransformationContext &context) const {
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Squeeze>({ make_op_label<opset1::Multiply>(), make_op_label<opset1::Constant>() }));
    addPattern(
        pass,
        context,
        make_op_pattern<opset1::Squeeze>({ make_op_label<opset1::Multiply>() }));
}

static std::shared_ptr<ngraph::Node> MyFold(const std::shared_ptr<Node>& operation, const std::shared_ptr<Node>& newConstant) {
    auto input = operation->input_values();
    input[0] = newConstant;
    auto newOp = operation->clone_with_new_inputs(input);
    OutputVector foldResult;
    newOp->constant_fold(foldResult);
    return foldResult[0].get_node_shared_ptr();
}

void SqueezeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!LayerTransformation::canBeTransformed(context, m.get_match_root())) {
        return;
    }
    const std::shared_ptr<Node> squeeze = separateInStandaloneBranch(m.get_match_root());
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(squeeze);
    std::shared_ptr<ngraph::Node> multiplyConstantNode = dequantization.multiply->get_argument(1);
    if (dequantization.multiply != nullptr && multiplyConstantNode->get_shape() == dequantization.data->get_shape()
        && multiplyConstantNode->get_shape().size() > 1) {
        auto newConstant = MyFold(squeeze, multiplyConstantNode);
        dequantization.multiply->set_argument(1, newConstant);
    }
    std::shared_ptr<ngraph::Node> subtractConstantNode = dequantization.subtract->get_argument(1);

    if (dequantization.subtract != nullptr && subtractConstantNode->get_shape() == dequantization.data->get_shape()
        && subtractConstantNode->get_shape().size() > 1) {
        auto newConstant = MyFold(squeeze, subtractConstantNode);
        dequantization.subtract->set_argument(1, newConstant);

    }

    moveDequantizationAfter(context, squeeze, dequantization, true);
}

//bool SqueezeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
//    return false;
//}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
