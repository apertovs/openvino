// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

#include "transformations/low_precision/layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API FuseFakeQuantizeTransformation : public LayerTransformation {
public:
    FuseFakeQuantizeTransformation(const Params& params) : LayerTransformation(params) {}
    ~FuseFakeQuantizeTransformation() override {}
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;

private:
    std::shared_ptr<opset1::FakeQuantize> handleDequantization(
        TransformationContext& context,
        const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const;

    std::shared_ptr<opset1::FakeQuantize> handleAdd(
        TransformationContext& context,
        const std::shared_ptr<opset1::FakeQuantize>& fakeQuantize) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
