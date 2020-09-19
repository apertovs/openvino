// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/avg_pool.hpp>
#include <transformations/low_precision/max_pool.hpp>
#include <transformations/low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/avg_pool_function.hpp"

using namespace testing;
using namespace ngraph::pass;

class AvgPoolTransformationTestValues {
public:
    class Actual {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantization;
    };

    class Expected {
    public:
        ngraph::element::Type precisionBeforeDequantization;
        ngraph::builder::subgraph::DequantizationOperations dequantizationBefore;
        ngraph::element::Type precisionAfterOperation;
        ngraph::builder::subgraph::DequantizationOperations dequantizationAfter;
    };
    ngraph::pass::low_precision::LayerTransformation::Params params;
    Actual actual;
    Expected expected;
};

typedef std::tuple<
    ngraph::Shape,
    bool, // additional FakeQuantize After
    std::string, // additional layer before FQ
    AvgPoolTransformationTestValues> AvgPoolTransformationParams;

class AvgPoolTransformation : public LayerTransformation, public testing::WithParamInterface<AvgPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::Shape shape = std::get<0>(GetParam());
        const bool addFQ = std::get<1>(GetParam());
        const std::string additionalLayer = std::get<2>(GetParam());
        const AvgPoolTransformationTestValues testValues = std::get<3>(GetParam());

        actualFunction = ngraph::builder::subgraph::AvgPoolFunction::getOriginal(
            shape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization,
            addFQ,
            additionalLayer);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::AvgPoolTransformation, ngraph::opset1::AvgPool>(testValues.params);
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::AvgPoolFunction::getReference(
            shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter,
            addFQ,
            additionalLayer);
    }

    static std::string getTestCaseName(testing::TestParamInfo<AvgPoolTransformationParams> obj) {
        const ngraph::Shape shape = std::get<0>(obj.param);
        const bool addFQ = std::get<1>(obj.param);
        const std::string additionalLayer = std::get<2>(obj.param);
        const AvgPoolTransformationTestValues testValues = std::get<3>(obj.param);
        std::ostringstream result;
        result <<
            shape << "_" <<
            addFQ << "_" <<
            additionalLayer << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            testValues.expected.precisionAfterOperation << "_" <<
            testValues.expected.precisionBeforeDequantization;
        return result.str();
    }
};

TEST_P(AvgPoolTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<std::string> additionalLayer = {
    "",
    "maxpool"  // any transparent layer
};

const std::vector<bool> addFQ = {
    true,
    false
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<AvgPoolTransformationTestValues> testValues = {
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false), // Layer params
        /* Actual */
        {
            ngraph::element::u8, // Precision before dequantization
            /* Dequantization */
            {
                {ngraph::element::f32}, // Convert
                { 128.0f }, // Subtract
                { 0.02f } // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::u8, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                { ngraph::element::f32 }, // Convert
                { 128.0f }, // Subtract
                { 0.02f } // Multiply
            }
        }
    },
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ngraph::element::i8, // Precision before dequantization
            /* Dequantization */
            {
                {ngraph::element::f32}, // Convert
                {}, // Subtract
                { 0.02f } // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::i8, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {ngraph::element::f32}, // Convert
                {}, // Subtract
                { 0.02f } // Multiply
            }
        }
    },
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(false).setUpdatePrecisions(false), // Layer params

        /* Actual */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                { 128.0f }, // Subtract
                { 0.02f } // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                { 128.0f }, // Subtract
                { 0.02f } // Multiply
            }
        }
    },
    {
        LayerTransformation::createParamsU8I8().setSupportAsymmetricQuantization(true), // Layer params

        /* Actual */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        },
        /* Expected */
        {
            ngraph::element::f32, // Precision before dequantization
            /* Dequantization before */
            {},
            ngraph::element::f32, // Precision after dequantization
            /* Dequantization after */
            {
                {}, // Convert
                {0.5f}, // Subtract
                {2.0f} // Multiply
            }
        }
    },
};

INSTANTIATE_TEST_CASE_P(
    LPT,
    AvgPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(addFQ),
        ::testing::ValuesIn(additionalLayer),
        ::testing::ValuesIn(testValues)),
    AvgPoolTransformation::getTestCaseName);
