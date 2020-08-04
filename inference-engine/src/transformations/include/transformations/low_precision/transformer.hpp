// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph_ops/type_relaxed.hpp>

#include "layer_transformation.hpp"
#include "iparams_manager.hpp"
#include "ilayer_transformations_manager.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

class TRANSFORMATIONS_API LowPrecisionTransformations {
public:
    LowPrecisionTransformations() {}
    LowPrecisionTransformations(
        const std::map<std::string, LayerTransformationPtr>& branchSpecificTransformations,
        const std::map<std::string, LayerTransformationPtr>& transformations,
        const std::map<std::string, LayerTransformationPtr>& cleanupTransformations);

    void setUpdatePrecisions(const bool updatePrecisions);
    void setQuantizedTensorAlignmentOnActivations(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnActivations);
    void setQuantizedTensorAlignmentOnWeights(const LayerTransformation::QuantizedTensorAlignment quantizedTensorAlignmentOnWeights);
    LowPrecisionTransformations& remove(const std::string& operationType);
    LowPrecisionTransformations& removeBranchSpecificTransformations(const std::string& operationType);
    LowPrecisionTransformations& removeTransformations(const std::string& operationType);
    LowPrecisionTransformations& removeCleanupTransformations(const std::string& operationType);

    /**
     * Add branch specific transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addBranchSpecific(const LayerTransformation::Params& params) {
        const std::string typeName = typeid(ngraph::op::TypeRelaxed<Operation>).name();

        const auto it = branchSpecificTransformations.find(typeName);
        if (it != branchSpecificTransformations.end()) {
            branchSpecificTransformations.erase(it);
        }

        branchSpecificTransformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    /**
     * Add transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& add(const LayerTransformation::Params& params) {
        const std::string typeName = typeid(ngraph::op::TypeRelaxed<Operation>).name();

        const auto it = transformations.find(typeName);
        if (it != transformations.end()) {
            transformations.erase(it);
        }

        transformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    /**
     * Add cleanup transformation. Transformation type and operation type are required.
     * Operation type is used to find transformation by operation during precision definition.
     */
    template <class Transformation, class Operation>
    LowPrecisionTransformations& addCleanup(const LayerTransformation::Params& params) {
        const std::string typeName = typeid(ngraph::op::TypeRelaxed<Operation>).name();

        const auto it = cleanupTransformations.find(typeName);
        if (it != cleanupTransformations.end()) {
            cleanupTransformations.erase(it);
        }

        cleanupTransformations.emplace(typeName, std::make_shared<Transformation>(params));
        return *this;
    }

    template <class Operation>
    static std::string getType() {
        return typeid(ngraph::op::TypeRelaxed<Operation>).name();
    }

    static std::string getType(const Node& operation) {
        return typeid(operation).name();
    }

    LayerTransformationPtr find(const std::string& transformationName) const;

    template <class Operation>
    LayerTransformationPtr find() const {
        const std::string transformationKey = getType<Operation>();
        auto it = branchSpecificTransformations.find(transformationKey);
        if (it != branchSpecificTransformations.end()) {
            return it->second;
        }

        it = transformations.find(transformationKey);
        if (it != transformations.end()) {
            return it->second;
        }

        it = cleanupTransformations.find(transformationKey);
        if (it != cleanupTransformations.end()) {
            return it->second;
        }

        return nullptr;
    }

    void setParamsManager(IParamsManager* paramsManager) noexcept;
    void setLayerTransformationsManager(ILayerTransformationsManager* layerTransformationsManager) noexcept;

    // Key is not a layer type, but just a name of transformation
    // Layer type (or a pattern) is defined by transformation itself as an ngraph matcher
    std::map<std::string, LayerTransformationPtr> branchSpecificTransformations;
    std::map<std::string, LayerTransformationPtr> transformations;
    std::map<std::string, LayerTransformationPtr> cleanupTransformations;

private:
    static void setParamsManager(IParamsManager* paramsManager, std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
    static void setLayerTransformationsManager(
        ILayerTransformationsManager* layerTransformationsManager,
        std::map<std::string, LayerTransformationPtr>& transformations) noexcept;
};

/**
 * @brief low precision transformation component.
  */
class TRANSFORMATIONS_API LowPrecisionTransformer : public IParamsManager, ILayerTransformationsManager {
public:
    static LowPrecisionTransformations getAllTransformations(const LayerTransformation::Params& params = LayerTransformation::Params());

    LowPrecisionTransformer();
    LowPrecisionTransformer(const LowPrecisionTransformations& transformations);
    void transform(std::shared_ptr<Function> network);

    // IParamsManager interface implementation
    std::vector<element::Type> getPrecisionsOnActivations(const Node& op) const noexcept override;

    // ILayerTransformationsManager interface implementation
    bool isQuantized(const std::shared_ptr<Node>& layer) const noexcept override;
    bool isPrecisionPreserved(const std::shared_ptr<Node>& layer) const noexcept override;

private:
    LowPrecisionTransformations transformations;

    void registerAllMatchers(
        std::map<std::string, LayerTransformationPtr> transformations,
        GraphRewrite& pass,
        TransformationContext& context);
};

class TRANSFORMATIONS_API TypeRelaxedReplacer : public GraphRewrite {
public:
    TypeRelaxedReplacer();
};

} // namespace low_precision
} // namespace pass
} // namespace ngraph
