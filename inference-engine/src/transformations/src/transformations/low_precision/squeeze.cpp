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

void SqueezeTransformation::transform(TransformationContext& context, ngraph::pattern::Matcher &m) const {
    if (!LayerTransformation::canBeTransformed(context, m.get_match_root())) {
        return;
    }

    const std::shared_ptr<Node> squeeze = separateInStandaloneBranch(m.get_match_root());
    FakeQuantizeDequantization dequantization = NetworkHelper::getDequantization(squeeze);
    //if (dequantization.multiply != nullptr) {
    //    auto fold_result = fold<opset1::Squeeze>(dequantization.multiply, squeeze->get_argument(1));
    //    dequantization.multiply = as_type_ptr<ngraph::op::v1::Multiply>(fold_result);
    //}

    //if (dequantization.subtract != nullptr) {
    //    auto fold_result = fold<opset1::Squeeze>(dequantization.subtract, squeeze->get_argument(1));
    //    dequantization.subtract = as_type_ptr<ngraph::op::v1::Subtract>(fold_result);
    //}

    //if (dequantization.convert != nullptr) {
    //    auto fold_result = fold<opset1::Squeeze>(dequantization.convert, squeeze->get_argument(1));
    //    dequantization.convert = as_type_ptr<ngraph::op::v0::Convert>(fold_result);
    //}
    moveDequantizationAfter<opset1::Squeeze>(context, squeeze, dequantization, false);
}

//bool SqueezeTransformation::isPrecisionPreserved(std::shared_ptr<Node> layer) const noexcept {
//    return false;
//}

} // namespace low_precision
} // namespace pass
} // namespace ngraph
