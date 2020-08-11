// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include <transformations_visibility.hpp>

#include "ngraph/op/op.hpp"

namespace ngraph {
namespace op {

class TRANSFORMATIONS_API PowerIE : public Op {
public:
    static constexpr NodeTypeInfo type_info{"PowerIE", 1};
    const NodeTypeInfo& get_type_info() const override { return type_info; }

    PowerIE(const Output<Node>& data_batch,
            const float power, const float scale, const float shift, const element::Type output_type = element::undefined);

    void validate_and_infer_types() override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    void set_output_type(size_t i,
        const element::Type& element_type,
        const PartialShape& pshape) override;

    float scale, power, shift;

private:
    element::Type m_output_type;
};

}  // namespace op
}  // namespace ngraph
