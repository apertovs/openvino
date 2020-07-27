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

using ngraph::builder::subgraph::SqueezeFunction;


class SqueezeTransformationTestValues {
public:
    low_precision::LayerTransformation::Params params;
    SqueezeFunction::LayerDescription multiplyActual;
    SqueezeFunction::LayerDescription multiplyExpected;
    SqueezeFunction::LayerDescription subtractActual;
    SqueezeFunction::LayerDescription subtractExpected;
    std::vector<size_t> squeezeArgs;
    ngraph::Shape shapeBefore;
    ngraph::Shape shapeAfter;
    ngraph::element::Type precision;
};


class SqueezeTransformation : public LayerTransformation, public testing::WithParamInterface<SqueezeTransformationTestValues> {
public:
    void SetUp() override {
        const SqueezeTransformationTestValues testValues = GetParam();
        const ngraph::element::Type precision = testValues.precision;

        actualFunction = ngraph::builder::subgraph::SqueezeFunction::getOriginal(
            precision,
            testValues.shapeBefore,
            testValues.squeezeArgs,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.multiplyActual,
                testValues.subtractActual
            });

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::SqueezeTransformation, ngraph::opset1::Squeeze>(testValues.params);

        VisualizeTree("C:\\result\\squeeze.before").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });


        transform.transform(actualFunction);        

        VisualizeTree("C:\\result\\squeeze.after").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ actualFunction });


        referenceFunction = ngraph::builder::subgraph::SqueezeFunction::getReference(
            precision,
            testValues.shapeBefore,
            testValues.squeezeArgs,
            {
                testValues.params.updatePrecisions ? testValues.params.precisionsOnActivations[0] : precision,
                testValues.multiplyExpected,
                testValues.subtractExpected
            });
        VisualizeTree("C:\\result\\reference.result").run_on_module(std::vector<std::shared_ptr<ngraph::Function>>{ referenceFunction });

    }

    static std::string getTestCaseName(testing::TestParamInfo<SqueezeTransformationTestValues> obj) {
        const SqueezeTransformationTestValues testValues = obj.param;

        std::ostringstream result;
        result <<
            testValues.multiplyActual.shape << "_" <<
            testValues.multiplyExpected.shape << "_" <<
            testValues.subtractActual.shape << "_" <<
            testValues.subtractExpected.shape << "_" <<
            testValues.shapeBefore << "_" <<
            testValues.shapeAfter;

        return result.str();
    }
};

TEST_P(SqueezeTransformation, CompareFunctions) {
    InitNodeInfo().run_on_function(actualFunction);
    actualFunction->validate_nodes_and_infer_types();


    auto res = compare_functions(referenceFunction, actualFunction, true);
  
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<SqueezeTransformationTestValues> testValues = {
    /*
    params;
    multiplyActual(@values, @shape);
    multiplyExpected(@values, @shape);
    subtractActual(@values, @shape);
    subtractExpected(@values, @shape);
    squeezeArgs;
    shapeBefore;
    shapeAfter
    precision
    */

    //{
    //    LayerTransformation::createParamsU8I8(),
    //    { { 0.2f }, { 1, 1000, 1, 1 } },
    //    { { 0.2f }, { 1, 1000 } },
    //    { { 128 }, { 1, 1000, 1, 1 } },
    //    { { 128 }, { 1, 1000 } },
    //    { 2u, 3u },
    //    { 1, 1000, 1, 1 },
    //    { 1, 1000 },
    //    ngraph::element::f32
    //},

    //{
    //    LayerTransformation::createParamsU8I8(),
    //    { { 0.5f }, { 1 } },
    //    { { 0.5f }, { 1 } },
    //    { { 32 }, { 1000, 1, 1, 1 } },
    //    { { 32 }, { 1000 } },
    //    { 1, 2, 3 },
    //    { 1000, 1, 1, 1 },
    //    { 1000 },
    //    ngraph::element::f32
    //},
    //{
    //    LayerTransformation::createParamsI8I8(),
    //    { { 0.1f }, { 1 } },
    //    { { 0.1f }, { 1 } },
    //    { { 256 }, { 1 } },
    //    { { 256 }, { 1 } },
    //    { 1u, 3u },
    //    { 1, 1, 1000, 1 },
    //    { 1, 1000 },
    //    ngraph::element::f32
    //},
    {
        LayerTransformation::createParamsI8I8(),
        { { 0.2f }, { 1, 1000, 1, 1 } },
        { { 0.2f }, { 1000 } },
        { { 128 }, { 1, 1000, 1, 1 } },
        { { 128 }, { 1000 } },
        { },
        { 1, 1000, 1, 1 },
        { 1000 },
        ngraph::element::f32
    }
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    SqueezeTransformation,
    ::testing::ValuesIn(testValues),
    SqueezeTransformation::getTestCaseName);
