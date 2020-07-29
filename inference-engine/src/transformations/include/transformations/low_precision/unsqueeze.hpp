// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include "layer_transformation.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API UnsqueezeTransformation : public LayerTransformation {
public:
    UnsqueezeTransformation(const Params& params);
    void registerMatcherIn(GraphRewrite& pass, TransformationContext& context) const override;
    void transform(TransformationContext& context, ngraph::pattern::Matcher &m) const override;
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
