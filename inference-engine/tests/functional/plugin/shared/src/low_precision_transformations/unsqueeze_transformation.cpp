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
#include "low_precision_transformations/unsqueeze_transformation.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "ngraph_functions/low_precision_transformations/unsqueeze_function.hpp"

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

InferenceEngine::Blob::Ptr UnsqueezeTransformation::GenerateInput(const InferenceEngine::InputInfo &info) const {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    UnsqueezeTransformationParam unsqueezeParam;
    std::string targetDevice;

    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = unsqueezeParam.fakeQuantize;

    return FuncTestUtils::createAndFillBlobConsistently(
        info.getTensorDesc(),
        static_cast<uint32_t>(fqOnData.empty() ? 25.f : fqOnData.outputHighValues[0] - fqOnData.outputLowValues[0]),
        static_cast<int32_t>(fqOnData.empty() ? -12.5f : fqOnData.outputLowValues[0]),
        1ul);
}

std::string UnsqueezeTransformation::getTestCaseName(testing::TestParamInfo<UnsqueezeTransformationParams> obj) {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::string targetDevice;
    UnsqueezeTransformationParam unsqueezeParam;
    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = obj.param;
    std::ostringstream result;
    result << unsqueezeParam.shape <<  "_" <<
        targetDevice << "_" <<
        version << "_" <<
        unsqueezeParam.fakeQuantize << "_" <<
        unsqueezeParam.unsqueezeAxes << "_" <<
        netPrecision << "_" <<
        params.updatePrecisions << "_" <<
        unsqueezeParam.shape;
    return result.str();
}
void UnsqueezeTransformation::SetUp() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    UnsqueezeTransformationParam unsqueezeParam;

    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = this->GetParam();

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::UnsqueezeFunction::getOriginal(
        netPrecision,
        unsqueezeParam.shape,
        unsqueezeParam.fakeQuantize,
        unsqueezeParam.unsqueezeAxes);

    ngraph::pass::InitNodeInfo().run_on_function(function);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void UnsqueezeTransformation::validate() {
    ngraph::element::Type netPrecision;
    ngraph::pass::low_precision::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::string targetDevice;
    UnsqueezeTransformationParam unsqueezeParam;
    std::tie(netPrecision, targetDevice, params, version, unsqueezeParam) = this->GetParam();

    const ngraph::builder::subgraph::FakeQuantizeOnData& fqOnData = unsqueezeParam.fakeQuantize;

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
        EXPECT_EQ("Unsqueeze", outputLayer->type);
    } else {
        EXPECT_EQ("ScaleShift", outputLayer->type);

        EXPECT_EQ(1ul, outputLayer->insData.size());
        const InferenceEngine::DataPtr insData = outputLayer->insData[0].lock();
        EXPECT_TRUE(insData != nullptr);
        const InferenceEngine::CNNLayerPtr unsqueeze = getCreatorLayer(insData).lock();
        EXPECT_TRUE(unsqueeze != nullptr);
        EXPECT_EQ("Unsqueeze", unsqueeze->type);

        if (params.updatePrecisions) {
            const InferenceEngine::Precision precision = unsqueeze->outData[0]->getTensorDesc().getPrecision();
            EXPECT_TRUE((precision == InferenceEngine::Precision::U8) || (precision == InferenceEngine::Precision::I8));
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(UnsqueezeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
