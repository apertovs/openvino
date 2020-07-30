// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <memory>

#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"

namespace LayerTestsDefinitions {

class SqueezeTransformationParam {
public:
    ngraph::builder::subgraph::FakeQuantizeOnData fakeQuantize;
    std::vector<float> squeezeAxes;
    ngraph::Shape shape;
};

typedef std::tuple<
    ngraph::element::Type,
    std::string,
    ngraph::pass::low_precision::LayerTransformation::Params,
    LayerTestsUtils::LayerTransformation::LptVersion,
    SqueezeTransformationParam
> SqueezeTransformationParams;

class SqueezeTransformation :
    public testing::WithParamInterface<SqueezeTransformationParams>,
    public LayerTestsUtils::LayerTransformation {
public:
    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override;
    static std::string getTestCaseName(testing::TestParamInfo<SqueezeTransformationParams> obj);

protected:
    void SetUp() override;

private:
    void validate();
};

}  // namespace LayerTestsDefinitions
