// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/squeeze.hpp>
#include <transformations/low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/squeeze_function.hpp"

#include <ngraph/pass/visualize_tree.hpp>

using namespace testing;
using namespace ngraph::pass;

class SqueezeTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

typedef std::tuple<
    ngraph::element::Type,
    ngraph::Shape,
    SqueezeTransformationTestValues> SqueezeTransformationParams;

class SqueezeTransformation : public LayerTransformation, public testing::WithParamInterface<SqueezeTransformationParams> {
public:
    void SetUp() override {
        const ngraph::element::Type precision = std::get<0>(GetParam());
        const ngraph::Shape shape = std::get<1>(GetParam());
        const SqueezeTransformationTestValues testValues = std::get<2>(GetParam());

        actualFunction = ngraph::builder::subgraph::SqueezeFunction::getOriginal(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });
        
        VisualizeTree("C:\\result\\squeeze.before").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });


        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::SqueezeTransformation, ngraph::opset1::Squeeze>(testValues.params);
        
        transform.transform(actualFunction);
        VisualizeTree("C:\\result\\squeeze.after").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });

        

        referenceFunction = ngraph::builder::subgraph::SqueezeFunction::getReference(
            precision,
            shape,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.subtractValues,
                testValues.mutliplyValues
            });
        VisualizeTree("C:\\result\\reference.before").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });

    }

    static std::string getTestCaseName(testing::TestParamInfo<SqueezeTransformationParams> obj) {
        const ngraph::element::Type precision = std::get<0>(obj.param);
        const ngraph::Shape shape = std::get<1>(obj.param);
        const SqueezeTransformationTestValues testValues = std::get<2>(obj.param);
        return LayerTransformation::getTestCaseNameByParams(precision, shape, testValues.params);
    }
};

TEST_P(SqueezeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();


    auto res = compare_functions(referenceFunction, actualFunction, true);
  
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 1000, 1, 1 }
};

const std::vector<SqueezeTransformationTestValues> testValues = {
    { LayerTransformation::createParamsU8I8(), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsU8I8().setUpdatePrecisions(false), { 128 }, { 0.02f } },
    { LayerTransformation::createParamsI8I8(), { 128 }, { 0.02f } },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    SqueezeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    SqueezeTransformation::getTestCaseName);
