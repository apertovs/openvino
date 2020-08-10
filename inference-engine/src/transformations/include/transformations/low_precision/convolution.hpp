// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include "weightable_layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API ConvolutionTransformation : public WeightableLayerTransformation {
public:
    ConvolutionTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    bool transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
    void transform2(TransformationContext& context, ngraph::pattern::Matcher &m) const;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
