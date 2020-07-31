// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <ngraph/ngraph.hpp>
#include "ngraph_ops/type_relaxed.hpp"

#include "transformations/low_precision/network_helper.hpp"

#include "ngraph_functions/low_precision_transformations/common/add.hpp"
#include "ngraph_functions/low_precision_transformations/common/fake_quantize_on_data.hpp"
#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

template <typename Operation, typename OperationDesc>
std::shared_ptr<Node> makeElementwise(const std::shared_ptr<ngraph::Node> data, const OperationDesc& description) {
    std::vector<size_t> shape;
    if (description.constantShapeIsDefined) {
        shape = description.constantShape;
    } else {
        if (description.values.size() == 1ul) {
            shape = std::vector<size_t>({});
        } else {
            shape = std::vector<size_t>(data->get_output_shape(0).size(), 1ul);
            shape[shape.size() >= 2 ? 1ul : 0] = description.values.size();
        }
    }

    const auto operationConst = std::make_shared<ngraph::opset1::Constant>(data->get_output_element_type(0), shape, description.values);

    std::shared_ptr<Operation> operation;
    if ((description.outPrecision == element::undefined) || (description.outPrecision == data->get_output_element_type(0))) {
        operation = std::make_shared<Operation>(data, operationConst);
    } else {
        operation = std::make_shared<op::TypeRelaxed<Operation>>(data, operationConst);
        ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(operation, description.outPrecision);
    }

    return operation;
}

std::shared_ptr<Node> makeDequantization(
    const std::shared_ptr<ngraph::Node> data,
    const DequantizationOperations& dequantizationOperations);

std::shared_ptr<ngraph::opset1::FakeQuantize> makeFakeQuantize(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

// TODO: refactor
std::shared_ptr<Node> makeFakeQuantizeTypeRelaxed(
    const std::shared_ptr<ngraph::Node>& input,
    const ngraph::element::Type precision,
    const FakeQuantizeOnData& fqOnData);

} // namespace subgraph
} // namespace builder
} // namespace ngraph
