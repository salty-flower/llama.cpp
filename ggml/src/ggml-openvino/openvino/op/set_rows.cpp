#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto data = context.get_input(0);
    auto indices = context.get_input(1);
    auto dst = context.get_input(2);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    auto dst_shape = context.get_output_shape().to_shape();

    auto ind_squeezed =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
    auto data_reshaped = std::make_shared<ov::op::v1::Reshape>(
        data,
        ov::op::v0::Constant::create(ov::element::i64, {4},
                                     {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) dst_shape[3]}),
        false);
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    Output<Node> res;
    bool scatter_reshaped_dst = false;
    if (context.is_stateful()) {
        int concat_axis = 1;
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else {
        if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
            auto dst_partial = dst.get_partial_shape();
            auto data_shape = context.get_input_shape(0).to_shape();
            auto index_shape = context.get_input_shape(1).to_shape();
            const auto data_update_len = data_shape[2] * data_shape[3];
            const auto index_len = index_shape[3];
            if (dst_partial.rank().is_static() && dst_partial.size() == 4 && dst_partial[3].is_static() &&
                dst_partial[3].get_length() == 1 && index_len > 0 && data_update_len % index_len == 0) {
                auto lane_width = data_update_len / index_len;
                Output<Node> ind_flattened = ind_squeezed;
                if (lane_width > 1) {
                    auto ind_expanded = std::make_shared<ov::op::v1::Multiply>(
                        std::make_shared<ov::op::v0::Unsqueeze>(
                            ind_squeezed, ov::op::v0::Constant::create(ov::element::i64, {1}, {1})),
                        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {lane_width}));
                    auto lane_offsets = std::make_shared<ov::op::v0::Unsqueeze>(
                        std::make_shared<ov::op::v4::Range>(
                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {lane_width}),
                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1}), ov::element::i64),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
                    ind_flattened = std::make_shared<ov::op::v1::Reshape>(
                        std::make_shared<ov::op::v1::Add>(ind_expanded, lane_offsets),
                        ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}), false);
                }
                auto data_flattened = std::make_shared<ov::op::v1::Reshape>(
                    data,
                    ov::op::v0::Constant::create(ov::element::i64, {4},
                                                 {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) 1}),
                    false);
                res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_flattened, data_flattened, axes);
                scatter_reshaped_dst = true;
            }
        }
        if (!scatter_reshaped_dst) {
            res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
        }
    }

    if (!scatter_reshaped_dst) {
        if (auto dst_reshape = std::dynamic_pointer_cast<ov::op::v1::Reshape>(dst.get_node_shared_ptr())) {
            // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
            // ctx_per_seq is not fixed due to llama-bench compatibility
            auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
            std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                              dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                              dst_shape_partial[3].get_length()};
            res = std::make_shared<ov::op::v1::Reshape>(
                res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape), false);
        }
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
