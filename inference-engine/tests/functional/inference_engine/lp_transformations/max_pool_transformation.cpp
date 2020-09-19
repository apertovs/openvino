// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layer_transformation.hpp"

#include <string>
#include <memory>

#include <gtest/gtest.h>

#include <transformations/utils/utils.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/low_precision/max_pool.hpp>
#include <transformations/low_precision/transformer.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"
#include "simple_low_precision_transformer.hpp"
#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

using namespace testing;
using namespace ngraph::pass;

class MaxPoolTransformationTestValues {
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
    MaxPoolTransformationTestValues> MaxPoolTransformationParams;

class MaxPoolTransformation : public LayerTransformation, public testing::WithParamInterface<MaxPoolTransformationParams> {
public:
    void SetUp() override {
        const ngraph::Shape shape = std::get<0>(GetParam());
        const MaxPoolTransformationTestValues testValues = std::get<1>(GetParam());

        actualFunction = ngraph::builder::subgraph::MaxPoolFunction::getOriginal(
            shape,
            testValues.actual.precisionBeforeDequantization,
            testValues.actual.dequantization);

        SimpleLowPrecisionTransformer transform;
        transform.add<ngraph::pass::low_precision::MaxPoolTransformation, ngraph::opset1::MaxPool>(testValues.params);
        transform.transform(actualFunction);

        referenceFunction = ngraph::builder::subgraph::MaxPoolFunction::getReference(
            shape,
            testValues.expected.precisionBeforeDequantization,
            testValues.expected.dequantizationBefore,
            testValues.expected.precisionAfterOperation,
            testValues.expected.dequantizationAfter);
    }

    static std::string getTestCaseName(testing::TestParamInfo<MaxPoolTransformationParams> obj) {
        const ngraph::Shape shape = std::get<0>(obj.param);
        const MaxPoolTransformationTestValues testValues = std::get<1>(obj.param);

        std::ostringstream result;
        result <<
            shape << "_" <<
            testValues.actual.precisionBeforeDequantization << "_" <<
            testValues.actual.dequantization << "_" <<
            testValues.expected.dequantizationBefore << "_" <<
            testValues.expected.dequantizationAfter << "_" <<
            testValues.expected.precisionAfterOperation << "_" <<
            testValues.expected.precisionBeforeDequantization;
        return result.str();
    }
};

TEST_P(MaxPoolTransformation, CompareFunctions) {
    actualFunction->validate_nodes_and_infer_types();

    auto res = compare_functions(referenceFunction, actualFunction, true, true, true);
    ASSERT_TRUE(res.first) << res.second;
}

const std::vector<ngraph::element::Type> precisions = {
    ngraph::element::f32,
    // ngraph::element::f16
};

const std::vector<ngraph::Shape> shapes = {
    { 1, 32, 72, 48 }
};

const std::vector<MaxPoolTransformationTestValues> testValues = {
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
            ngraph::element::u8, // Precision after dequantization
            /* Dequantization after */
            {
                {ngraph::element::f32}, // Convert
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
            ngraph::element::i8, // Precision after dequantization
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
    MaxPoolTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(shapes),
        ::testing::ValuesIn(testValues)),
    MaxPoolTransformation::getTestCaseName);
