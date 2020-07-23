// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include <ngraph/rt_info.hpp>

#include "transformation_context.hpp"
#include "quantization_details.hpp"
#include "transformations/utils/utils.hpp"
#include "common/fake_quantize_dequantization.hpp"

namespace ngraph {
namespace pass {
namespace low_precision {

/**
* @brief NetworkHelper class encapsulates manipulations with nGraph function.
*/
class TRANSFORMATIONS_API NetworkHelper {
public:
    // Return true if `type` can be castable to at least one of `type`
    static bool is_castable_to_one_of(NodeTypeInfo type, const std::unordered_set<NodeTypeInfo>& types);

    static std::vector<Input<Node>> consumer_inputs(std::shared_ptr<Node> node);

    // Collect and return a vector with all nodes that consumes any of the `node` output
    static std::vector<std::shared_ptr<Node>> consumers(std::shared_ptr<Node> node);

    static Shape alignShapeForChannelDim(const Shape& shape, Rank rank);

    // return true if at least one child uses layer on weights
    static bool onWeights(std::shared_ptr<Node> layer);

    template <typename OperationType>
    static void setOutDataPrecision(std::shared_ptr<OperationType>, const element::Type& precision);

    static size_t getOutputChannelsCount(std::shared_ptr<const Node> layer, bool isOnWeights = false);

    static std::vector<std::shared_ptr<Node>> getParentsRecursivelyExceptTypes(
        std::shared_ptr<Node> layer,
        const std::unordered_set<NodeTypeInfo>& exceptionLayerTypes = {},
        const int portIndex = -1);

    static size_t getInputChannelsCount(std::shared_ptr<Node> layer);

    static size_t getGroupsCount(std::shared_ptr<Node> layer);

    // Remove node by connecting its 0th input with 0th output
    static void removeLayer(std::shared_ptr<Node> node);

    static std::shared_ptr<opset1::Multiply> swapMultiplyAndAdd(std::shared_ptr<opset1::Add> addAfterMultiply);

    static bool isScalarLike(std::shared_ptr<opset1::Constant> constant);

    static std::shared_ptr<opset1::Constant> distillToScalar(std::shared_ptr<opset1::Constant> constant);

    static std::shared_ptr<Node> getConstantInput(std::shared_ptr<Node> node);

    // Optimizes the series of multiplies after a given output port
    static std::shared_ptr<ngraph::opset1::Multiply> optimizeMultipliesAfter(std::shared_ptr<Node> multiply);

    static std::shared_ptr<opset1::Constant> roundWithTolerance(std::shared_ptr<Node> node, element::Type target_type, float tolerance = 1e-5);

    static std::tuple<std::shared_ptr<Node>, std::shared_ptr<Node>> decomposeFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        const element::Type precision,
        const float min,
        const float max,
        const bool hasZeroPoint,
        const bool updatePrecision);

    static std::shared_ptr<opset1::FakeQuantize> updateFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        element::Type precision,
        float min,
        float max);

    static FakeQuantizeDequantization createDequantization(
        const float dequantizationScale,
        const float dequantizationShift,
        const ngraph::element::Type originalPrecision,
        const ngraph::Shape dataNodeOutputShape,
        element::Type precision,
        float min,
        float max);

    static FakeQuantizeDequantization createDequantizationFromFakeQuantize(
        std::shared_ptr<opset1::FakeQuantize> fq,
        element::Type precision,
        float min,
        float max);

    static FakeQuantizeDequantization getDequantization(const std::shared_ptr<Node> node, const size_t parentIndex = 0ul);

    static std::shared_ptr<Node> optimizeSubtract(std::shared_ptr<opset1::Subtract> add);

    static void moveDequantization(
        const std::shared_ptr<ngraph::Node> operation,
        const std::shared_ptr<ngraph::Node> dequantization,
        const std::shared_ptr<ngraph::Node> scalesConst = nullptr,
        const std::shared_ptr<ngraph::Node> shiftsConst = nullptr);

    class InsertDequantizationResult {
    public:
        InsertDequantizationResult(
            const std::shared_ptr<Node>& newOperation,
            const std::shared_ptr<Node>& lastDequantization) : newOperation(newOperation), lastDequantization(lastDequantization) {}

        std::shared_ptr<Node> newOperation;
        std::shared_ptr<Node> lastDequantization;
    };

    template<class Operation>
    static std::shared_ptr<ngraph::Node> MyFold(const std::shared_ptr<ngraph::Node>& operation, const std::shared_ptr<Node> new_tensor) {
        auto input = operation->input_values();
        input[0] = new_tensor;
        auto newOp = operation->clone_with_new_inputs(input);
        OutputVector foldResult;
        newOp->constant_fold(foldResult);
        return foldResult[0].get_node_shared_ptr();
    }

    template <class Operation = ngraph::Node>
    static InsertDequantizationResult moveDequantizationAfter(
            const std::shared_ptr<ngraph::Node>& operation,
            const FakeQuantizeDequantization& dequantization,
            const bool updatePrecision) {

        if (as_type_ptr<Operation>(operation) == nullptr) {
            // THROW_IE_LPT_EXCEPTION << "operation template parameter is invalid";
        }

        bool isFused = static_cast<bool>(std::dynamic_pointer_cast<ngraph::op::util::FusedOp>(operation) != nullptr);

        std::vector<Output<Node>> inputs(operation->get_input_size());
        for (size_t i = 0; i < operation->get_input_size(); ++i) {
            inputs[i] = operation->get_input_node_shared_ptr(i);
        }

        const size_t dequantizationIndex = getInputIndex(dequantization.multiply, operation);
        inputs[dequantizationIndex] = dequantization.data;

        std::shared_ptr<ngraph::Node> newOperation = operation->clone_with_new_inputs(inputs);
        newOperation->set_friendly_name(operation->get_friendly_name());

        const std::shared_ptr<ngraph::opset1::Convert> convert = updatePrecision ? dequantization.convert : nullptr;

        auto parent = newOperation;
        if (convert != nullptr) {
            parent = convert->clone_with_new_inputs({ parent });
        }
        if (dequantization.subtract != nullptr) {
            auto subtractConstant = dequantization.subtract->get_input_node_shared_ptr(1);
            if (isFused) {
                subtractConstant = MyFold<Operation>(operation, subtractConstant);
            }
            parent = std::make_shared<opset1::Subtract>(parent, subtractConstant);
        }
        if (dequantization.multiply != nullptr) {
            auto multiplyConstant = dequantization.multiply->get_input_node_shared_ptr(1);
            if (isFused) {
                multiplyConstant = MyFold<Operation>(operation, multiplyConstant);
            }
            parent = std::make_shared<opset1::Multiply>(parent, multiplyConstant);
        }

        //std::shared_ptr<opset1::Multiply> replacement = std::make_shared<opset1::Multiply>(
        //    dequantization.subtract ?
        //    (convert ?
        //        std::make_shared<opset1::Subtract>(
        //            convert->clone_with_new_inputs({ newOperation }),
        //            dequantization.subtract->get_input_node_shared_ptr(1)) :
        //        std::make_shared<opset1::Subtract>(
        //            newOperation,
        //            dequantization.subtract->get_input_node_shared_ptr(1))) :
        //            (convert ? convert->clone_with_new_inputs({ newOperation }) : newOperation),
        //    dequantization.multiply->get_input_node_shared_ptr(1));

        replace_node(operation, parent);

        if (updatePrecision) {
            NetworkHelper::setOutDataPrecision(newOperation, newOperation->get_input_element_type(0));
        }

        return InsertDequantizationResult(newOperation, parent);
    }

    static size_t getInputIndex(const std::shared_ptr<ngraph::Node>& parent, const std::shared_ptr<ngraph::Node>& child);

    static std::vector<Output<Node>> getInputs(const std::shared_ptr<ngraph::Node>& node);

private:
    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(std::shared_ptr<Node> layer);
};

template <typename OperationType>
void NetworkHelper::setOutDataPrecision(std::shared_ptr<OperationType> layer, const element::Type& precision) {
    // check if it already exteded operation node
    if (auto relaxed_layer = std::dynamic_pointer_cast<ngraph::op::TypeRelaxedBase>(layer)) {
        relaxed_layer->set_overriden_output_type(precision);
        std::dynamic_pointer_cast<ngraph::Node>(layer)->validate_and_infer_types();
    } else {
        // TODO: Make such replacements in advance for all supported polymorphic layer types
        // extend a node with new semantics: overriden output data_type
        // FIXME: OperationType should be a real type of an object, otherwise it will lead to undefined behavior
        auto replacement = std::make_shared<ngraph::op::TypeRelaxed<OperationType>>(*layer, precision);
        copy_runtime_info(layer, replacement);
        replace_node(layer, replacement);
    }
}

template <typename T>
std::shared_ptr<Node> make_op_pattern(const ngraph::NodeVector& args) {
    return std::make_shared<ngraph::pattern::op::Any>(element::undefined, PartialShape{}, [](std::shared_ptr<Node> n) {return !!as_type_ptr<T>(n); }, args);
}

template <typename T>
std::shared_ptr<Node> make_op_label() {
    return std::make_shared<ngraph::pattern::op::Label>(
            element::undefined,
            PartialShape{},
            [](std::shared_ptr<Node> n) {return !!as_type_ptr<T>(n); });
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold(Args&&... args) {
    auto node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->constant_fold(folded)) {
            return folded[0].get_node_shared_ptr();
        }
    }
    return node;
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold_reshape(Args&&... args) {
    std::shared_ptr<Node> node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->input_value(0).get_node_shared_ptr()->is_constant() && node->input_value(1).get_node_shared_ptr()->is_constant()) {
            return std::make_shared<opset1::Constant>(
                    node->get_input_element_type(0),
                    Shape(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr())->template cast_vector<size_t>()),
                    as_type_ptr<opset1::Constant>(node->input_value(0).get_node_shared_ptr())->get_data_ptr());
        }
    }
    return node;
}

template <typename T, typename... Args>
std::shared_ptr<Node> fold_fake_quantize(Args&&... args) {
    std::shared_ptr<Node> node = std::make_shared<T>(std::forward<Args>(args)...);
    if (node->get_output_size() == 1) {
        OutputVector folded;
        if (node->input_value(0).get_node_shared_ptr()->is_constant() &&
            node->input_value(1).get_node_shared_ptr()->is_constant() &&
            node->input_value(2).get_node_shared_ptr()->is_constant() &&
            node->input_value(3).get_node_shared_ptr()->is_constant() &&
            node->input_value(4).get_node_shared_ptr()->is_constant() &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(1).get_node_shared_ptr()), 0) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(2).get_node_shared_ptr()), 254) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(3).get_node_shared_ptr()), -127) &&
            op::util::constantIsEqualTo(as_type_ptr<opset1::Constant>(node->input_value(4).get_node_shared_ptr()), 127)) {
            return fold<opset1::Add>(node->input_value(0), node->input_value(3));
        }
    }
    return node;
}

}  // namespace low_precision
}  // namespace pass
}  // namespace ngraph
