// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "ngraph/op/op.hpp"
#include <transformations/init_node_info.hpp>
#include "low_precision_transformations/squeeze_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/squeeze_function.hpp"

namespace LayerTestsDefinitions {

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

InferenceEngine::Blob::Ptr SqueezeTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SqueezeTransformationParam squeezeParam;
    std::string targetDevice;

    std::tie(netPrecision, targetDevice, params, version, squeezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

std::string SqueezeTransformation::getTestCaseName(testing::TestParamInfo<SqueezeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::string targetDevice;
    SqueezeTransformationParam squeezeParam;
    std::tie(netPrecision, targetDevice, params, version, squeezeParam) = obj.param;
    std::ostringstream result;
    result << squeezeParam.shape <<  "_" <<
        targetDevice << "_" <<
        version << "_" <<
        squeezeParam.fakeQuantize << "_" <<
        squeezeParam.squeezeAxes << "_" <<
        netPrecision << "_" <<
        params.updatePrecisions << "_" <<
        squeezeParam.shape;
    return result.str();
}
void SqueezeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    SqueezeTransformationParam squeezeParam;

    std::tie(netPrecision, targetDevice, params, version, squeezeParam) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::SqueezeFunction::getOriginal(
        netPrecision,
        squeezeParam.shape,
        squeezeParam.fakeQuantize,
        squeezeParam.squeezeAxes);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void SqueezeTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::string targetDevice;
    SqueezeTransformationParam squeezeParam;
    std::tie(netPrecision, targetDevice, params, version, squeezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = squeezeParam.fakeQuantize;

    const auto paramsCNN = LayerTestsUtils::LayerTransformationParamsFactory::createParams();
    const InferenceEngine::CNNNetwork network = transform(paramsCNN);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);

    if (fqOnData.empty() ||
        (fqOnData.isSigned() && ((fqOnData.outputLowValues[0] / fqOnData.outputHighValues[0]) != (-128.f/127.f))) ||
        (!fqOnData.isSigned() && ((fqOnData.outputLowValues[0] != 0.f)))) {
        EXPECT_EQ("Squeeze", outputLayer->type);
    } else {
        EXPECT_EQ("ScaleShift", outputLayer->type);

        EXPECT_EQ(1ul, outputLayer->insData.size());
        const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
        EXPECT_TRUE(insData != nullptr);
        const InferenceEngine::CNNLayerPtr squeeze = getCreatorLayer(insData).lock();
        EXPECT_TRUE(squeeze != nullptr);
        EXPECT_EQ("Squeeze", squeeze->type);

        if (params.updatePrecisions) {
            const InferenceEngine::Precision precision = squeeze->outData[0]->getTensorDesc().getPrecision();
            EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(SqueezeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
