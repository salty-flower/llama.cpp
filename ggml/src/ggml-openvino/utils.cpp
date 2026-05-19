#include "utils.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino/ggml-decoder.h"
#include "ggml.h"
#include "openvino/frontend.h"
#include "openvino/input_model.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <openvino/core/any.hpp>
#include <openvino/core/graph_util.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/frontend/manager.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/compiled_model.hpp>
#include <openvino/runtime/infer_request.hpp>
#include <openvino/runtime/intel_npu/properties.hpp>
#include <openvino/runtime/properties.hpp>
#include <openvino/runtime/tensor.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

// Suppress  deprecation warning for ov::Tensor::data()
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static bool is_empty_graph_without_side_effects(const ggml_cgraph * cgraph) {
    if (cgraph->n_nodes == 0) {
        return true;
    }

    const ggml_tensor * output = cgraph->nodes[cgraph->n_nodes - 1];
    if (ggml_nelements(output) != 0) {
        return false;
    }

    for (int i = 0; i < cgraph->n_nodes; ++i) {
        if (cgraph->nodes[i]->op == GGML_OP_SET_ROWS) {
            return false;
        }
    }

    return true;
}

static std::string ov_shape_to_string(const ov::Shape & shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

static size_t ov_shape_size(const ov::Shape & shape) {
    size_t size = 1;
    for (const auto dim : shape) {
        size *= dim;
    }
    return size;
}

static bool is_view_like_output(const ggml_tensor * tensor) {
    return tensor->op == GGML_OP_VIEW ||
           tensor->op == GGML_OP_RESHAPE ||
           tensor->op == GGML_OP_PERMUTE ||
           tensor->op == GGML_OP_TRANSPOSE;
}

static std::vector<size_t> ov_row_major_strides(const ov::Shape & shape) {
    std::vector<size_t> strides(shape.size(), 1);
    for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    return strides;
}

static std::string sanitize_filename_part(const char * name) {
    std::string sanitized = name && name[0] != '\0' ? name : "unnamed";
    for (auto & ch : sanitized) {
        if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_') {
            ch = '_';
        }
    }
    if (sanitized.size() > 48) {
        sanitized.resize(48);
    }
    return sanitized;
}

static void maybe_dump_cgraph(const ggml_cgraph * cgraph) {
    const char * dump_dir = getenv("GGML_OPENVINO_DUMP_CGRAPH_DIR");
    if (dump_dir && dump_dir[0] != '\0') {
        static std::atomic<unsigned long long> dump_counter{0};
        const auto index = dump_counter.fetch_add(1);
        const auto first = cgraph->n_nodes > 0 ? sanitize_filename_part(cgraph->nodes[0]->name) : "empty";
        const auto last = cgraph->n_nodes > 0 ? sanitize_filename_part(cgraph->nodes[cgraph->n_nodes - 1]->name) : "empty";
        char filename[1024];
        snprintf(filename, sizeof(filename), "%s/cgraph_%06llu_n%d_%s__%s.txt",
                 dump_dir, index, cgraph->n_nodes, first.c_str(), last.c_str());
        std::string filename_string(filename);
        GgmlOvDecoder::dump_cgraph(cgraph, filename_string);
        return;
    }

    if (getenv("GGML_OPENVINO_DUMP_CGRAPH")) {
        std::string filename = "cgraph_ov.txt";
        GgmlOvDecoder::dump_cgraph(cgraph, filename);
    }
}

enum ggml_status ov_graph_compute(ggml_cgraph * cgraph, ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;
    try {
        maybe_dump_cgraph(cgraph);

        if (is_empty_graph_without_side_effects(cgraph)) {
            GGML_LOG_DEBUG("OpenVINO backend skipped empty graph without side effects\n");
            return GGML_STATUS_SUCCESS;
        }

        const auto is_static = ggml_openvino_is_npu() && getenv("GGML_OPENVINO_FORCE_DYNAMIC") == nullptr;

        GGML_ASSERT(ctx->runtime_context != nullptr);
        std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);

        return is_static ? ov_graph_compute_static(cgraph, r_ctx) : ov_graph_compute_dynamic(cgraph, r_ctx);
    } catch (const ov::Exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend ov::Exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (const std::exception & e) {
        GGML_LOG_ERROR("GGML OpenVINO backend std::exception: %s\n", e.what());
        return GGML_STATUS_FAILED;
    } catch (...) {
        GGML_LOG_ERROR("GGML OpenVINO backend unknown exception\n");
        return GGML_STATUS_FAILED;
    }
}

ov::Tensor create_ov_output_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                   std::shared_ptr<ov::InferRequest> infer_request,
                                   int output_index,
                                   const ggml_tensor * ggml_tensor) {
    auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
    ov::Shape output_shape;
    if (ggml_decoder->is_static()) {
        output_shape = infer_request->get_output_tensor(output_index).get_shape();
    } else {
        output_shape = ggml_decoder->get_shape(ggml_tensor);
    }

    ov::Tensor output_tensor(output_type, output_shape, ggml_tensor->data);
    return output_tensor;
}

static void copy_ov_output_to_ggml_strided(const ov::Tensor & output_tensor,
                                           const ggml_tensor * ggml_tensor,
                                           const ov::Shape & copy_shape,
                                           bool linear_source,
                                           size_t dst_axis = GGML_MAX_DIMS,
                                           size_t dst_axis_offset = 0) {
    const auto src_shape = output_tensor.get_shape();
    const auto dst_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    const size_t rank = copy_shape.size();
    GGML_ASSERT(rank > 0 && rank <= GGML_MAX_DIMS);

    const size_t element_size = output_tensor.get_byte_size() / output_tensor.get_size();
    const auto src_strides = ov_row_major_strides(src_shape);
    const auto * src = static_cast<const uint8_t *>(output_tensor.data());
    auto * dst = static_cast<uint8_t *>(ggml_tensor->data);

    size_t outer_size = 1;
    for (size_t i = 0; i + 1 < rank; ++i) {
        outer_size *= copy_shape[i];
    }
    const size_t inner_size = copy_shape[rank - 1];

    std::vector<size_t> idx(rank, 0);
    size_t linear_source_offset = 0;
    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t src_offset_elements = linear_source ? linear_source_offset : 0;
        size_t dst_offset_bytes = 0;
        for (size_t i = 0; i + 1 < rank; ++i) {
            if (!linear_source) {
                src_offset_elements += idx[i] * src_strides[i];
            }
            const size_t dst_idx = idx[i] + (i == dst_axis ? dst_axis_offset : 0);
            dst_offset_bytes += dst_idx * ggml_tensor->nb[rank - 1 - i];
        }

        std::memcpy(dst + dst_offset_bytes, src + src_offset_elements * element_size, inner_size * element_size);
        linear_source_offset += inner_size;

        for (int i = static_cast<int>(rank) - 2; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < copy_shape[i]) {
                break;
            }
            idx[i] = 0;
        }
    }
}

static bool is_last_two_axes_swapped(const ov::Shape & src_shape, const ov::Shape & dst_shape) {
    if (src_shape.size() != dst_shape.size() || src_shape.size() < 2) {
        return false;
    }

    const size_t rank = src_shape.size();
    for (size_t i = 0; i + 2 < rank; ++i) {
        if (src_shape[i] != dst_shape[i]) {
            return false;
        }
    }

    return src_shape[rank - 2] == dst_shape[rank - 1] &&
           src_shape[rank - 1] == dst_shape[rank - 2];
}

static void copy_ov_output_to_ggml_last_two_axes_swapped(const ov::Tensor & output_tensor,
                                                         const ggml_tensor * ggml_tensor,
                                                         size_t src_axis_a_len = SIZE_MAX,
                                                         size_t dst_axis_a_offset = 0) {
    const auto src_shape = output_tensor.get_shape();
    const auto dst_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    const size_t rank = src_shape.size();
    GGML_ASSERT(rank > 1 && rank <= GGML_MAX_DIMS);
    if (src_axis_a_len == SIZE_MAX) {
        src_axis_a_len = src_shape[rank - 2];
    }
    GGML_ASSERT(src_axis_a_len <= src_shape[rank - 2]);
    GGML_ASSERT(dst_axis_a_offset + src_axis_a_len <= dst_shape[rank - 1]);

    const size_t element_size = output_tensor.get_byte_size() / output_tensor.get_size();
    const auto src_strides = ov_row_major_strides(src_shape);
    const auto * src = static_cast<const uint8_t *>(output_tensor.data());
    auto * dst = static_cast<uint8_t *>(ggml_tensor->data);

    size_t outer_size = 1;
    for (size_t i = 0; i + 2 < rank; ++i) {
        outer_size *= src_shape[i];
    }

    std::vector<size_t> idx(rank, 0);
    const size_t src_axis_a = rank - 2;
    const size_t src_axis_b = rank - 1;
    const size_t dst_axis_a_nb = rank - 1 - src_axis_b;
    const size_t dst_axis_b_nb = rank - 1 - src_axis_a;
    for (size_t outer = 0; outer < outer_size; ++outer) {
        size_t src_outer_offset_elements = 0;
        size_t dst_outer_offset_bytes = 0;
        for (size_t i = 0; i + 2 < rank; ++i) {
            src_outer_offset_elements += idx[i] * src_strides[i];
            dst_outer_offset_bytes += idx[i] * ggml_tensor->nb[rank - 1 - i];
        }

        for (size_t a = 0; a < src_axis_a_len; ++a) {
            const size_t src_offset_elements = src_outer_offset_elements + a * src_strides[src_axis_a];
            const size_t dst_offset_bytes =
                dst_outer_offset_bytes + (dst_axis_a_offset + a) * ggml_tensor->nb[dst_axis_a_nb];
            for (size_t b = 0; b < src_shape[src_axis_b]; ++b) {
                std::memcpy(dst + dst_offset_bytes + b * ggml_tensor->nb[dst_axis_b_nb],
                            src + (src_offset_elements + b) * element_size,
                            element_size);
            }
        }

        for (int i = static_cast<int>(rank) - 3; i >= 0; --i) {
            idx[i]++;
            if (idx[i] < src_shape[i]) {
                break;
            }
            idx[i] = 0;
        }
    }
}

static void copy_ov_output_to_ggml(const ov::Tensor & output_tensor, const ggml_tensor * ggml_tensor) {
    const auto src_shape = output_tensor.get_shape();
    const auto dst_shape = GgmlOvDecoder::get_shape(ggml_tensor);

    if (src_shape == dst_shape) {
        copy_ov_output_to_ggml_strided(output_tensor, ggml_tensor, dst_shape, false);
        return;
    }

    if (is_last_two_axes_swapped(src_shape, dst_shape)) {
        copy_ov_output_to_ggml_last_two_axes_swapped(output_tensor, ggml_tensor);
        return;
    }

    if (output_tensor.get_size() == static_cast<size_t>(ggml_nelements(ggml_tensor))) {
        copy_ov_output_to_ggml_strided(output_tensor, ggml_tensor, dst_shape, true);
        return;
    }

    throw std::runtime_error("Unexpected output shape for " + std::string(ggml_tensor->name) +
                             ": src=" + ov_shape_to_string(src_shape) +
                             ", dst=" + ov_shape_to_string(dst_shape));
}

static bool can_bind_ov_output_directly(const ov::Shape & output_shape, const ggml_tensor * ggml_tensor) {
    return output_shape == GgmlOvDecoder::get_shape(ggml_tensor) &&
           ggml_is_contiguous(ggml_tensor) &&
           !is_view_like_output(ggml_tensor);
}

static bool is_static_cache_view_output(const ggml_tensor * tensor) {
    const std::string name(tensor->name);
    return name.rfind("cache_k_", 0) == 0 ||
           name.rfind("cache_v_", 0) == 0;
}

static void copy_static_prefill_output_to_ggml(const ov::Tensor & output_tensor,
                                               const ggml_tensor * ggml_tensor,
                                               size_t input_len,
                                               size_t chunk_size,
                                               int chunk_index) {
    const auto src_shape = output_tensor.get_shape();
    const auto dst_shape = GgmlOvDecoder::get_shape(ggml_tensor);
    if (src_shape.size() != dst_shape.size()) {
        throw std::runtime_error("Unexpected padded output rank for " + std::string(ggml_tensor->name));
    }

    if (output_tensor.get_size() == static_cast<size_t>(ggml_nelements(ggml_tensor))) {
        copy_ov_output_to_ggml(output_tensor, ggml_tensor);
        return;
    }

    const size_t valid_len = std::min(chunk_size, input_len - chunk_index * chunk_size);
    if (src_shape.size() == dst_shape.size() && src_shape.size() >= 2) {
        const size_t rank = src_shape.size();
        bool outer_dims_match = true;
        for (size_t i = 0; i + 2 < rank; ++i) {
            if (src_shape[i] != dst_shape[i]) {
                outer_dims_match = false;
                break;
            }
        }
        if (outer_dims_match &&
            src_shape[rank - 2] == chunk_size &&
            src_shape[rank - 1] == dst_shape[rank - 2] &&
            chunk_index * chunk_size + valid_len <= dst_shape[rank - 1]) {
            copy_ov_output_to_ggml_last_two_axes_swapped(
                output_tensor, ggml_tensor, valid_len, chunk_index * chunk_size);
            return;
        }
    }

    size_t axis = src_shape.size();
    size_t src_axis_len = 0;
    size_t dst_axis_len = 0;
    size_t dst_axis_offset = 0;
    for (size_t i = 0; i < src_shape.size(); ++i) {
        if (src_shape[i] == chunk_size && dst_shape[i] == valid_len) {
            axis = i;
            src_axis_len = chunk_size;
            dst_axis_len = valid_len;
            break;
        }
        if (src_shape[i] == chunk_size && dst_shape[i] == input_len) {
            axis = i;
            src_axis_len = chunk_size;
            dst_axis_len = valid_len;
            dst_axis_offset = chunk_index * chunk_size;
            break;
        }
    }
    if (axis == src_shape.size()) {
        for (size_t i = 0; i < src_shape.size(); ++i) {
            if (src_shape[i] > dst_shape[i]) {
                axis = i;
                src_axis_len = src_shape[i];
                dst_axis_len = dst_shape[i];
                break;
            }
        }
    }
    if (axis == src_shape.size()) {
        throw std::runtime_error("Unexpected padded output shape for " + std::string(ggml_tensor->name) +
                                 ": src=" + ov_shape_to_string(src_shape) +
                                 ", dst=" + ov_shape_to_string(dst_shape) +
                                 ", valid_len=" + std::to_string(valid_len) +
                                 ", chunk_size=" + std::to_string(chunk_size));
    }

    ov::Shape copy_shape = dst_shape;
    for (size_t i = 0; i < src_shape.size(); ++i) {
        if (i == axis) {
            copy_shape[i] = dst_axis_len;
        } else if (src_shape[i] != dst_shape[i]) {
            throw std::runtime_error("Unexpected padded output non-axis shape for " + std::string(ggml_tensor->name) +
                                     ": src=" + ov_shape_to_string(src_shape) +
                                     ", dst=" + ov_shape_to_string(dst_shape) +
                                     ", valid_len=" + std::to_string(valid_len) +
                                     ", chunk_size=" + std::to_string(chunk_size) +
                                     ", axis=" + std::to_string(axis));
        }
    }
    (void) src_axis_len;
    copy_ov_output_to_ggml_strided(output_tensor, ggml_tensor, copy_shape, false, axis, dst_axis_offset);
}

enum ggml_status ov_graph_compute_dynamic(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();
    const auto & config = ggml_openvino_get_compile_config();
    const auto & device = r_ctx->device;
    const auto & stateful = r_ctx->stateful;
    static auto is_static = false;

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    {
        std::shared_ptr<decoder_runtime_ctx> entry;
        ModelParams old_m_params;

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            auto it = r_ctx->decoder_cache.find(key);
            cache_hit = it != r_ctx->decoder_cache.end();
            if (cache_hit) {
                entry = it->second;
            } else {
                auto mutex = std::make_shared<std::mutex>();
                entry = std::make_shared<decoder_runtime_ctx>(mutex);
                r_ctx->decoder_cache[key] = entry;
            }
        }

        std::lock_guard<std::mutex> lock(*(entry->mutex));

        if (cache_hit) {
            ggml_decoder = entry->ptr;
            old_m_params = ggml_decoder->get_model_params();
            cache_hit = old_m_params.can_reuse_dynamically(m_params);
        }

        if (cache_hit) {
            std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
            ggml_decoder->set_compute_params(c_params);
            ggml_decoder->set_model_params(m_params);
            if (old_m_params.kv_buffer_changed(m_params)) {
                ggml_decoder->update_io(cgraph);
            }
            ggml_decoder->add_extra_inputs();
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                infer_request = r_ctx->infer_request_cache.at(key);
            }

            if (stateful) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                int32_t * pos_data = (int32_t *) inp_pos->data;
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                if (pos_data[0] == 0) {
                    infer_request->reset_state();
                    r_ctx->stateful_kv_size = pos_shape[3];
                } else if (r_ctx->stateful_kv_size == static_cast<size_t>(pos_data[0])) {
                    r_ctx->stateful_kv_size += pos_shape[3];
                } else {
                    auto states = infer_request->query_state();
                    for (auto state : states) {
                        auto state_tensor = state.get_state();
                        auto state_tensor_shape = state_tensor.get_shape();
                        if (static_cast<uint32_t>(pos_data[0]) > r_ctx->stateful_kv_size) {
                            std::string state_name;
                            try {
                                state_name = r_ctx->kv_state_input_name_map.at(state.get_name());
                            } catch (...) {
                                GGML_LOG_ERROR("GGML OpenVINO backend stateful inference failed: no input found for the state\n");
                                return GGML_STATUS_FAILED;
                            }
                            auto kv_tensor = get_ov_input_tensor(ggml_decoder, state_name);
                            kv_tensor.set_shape({state_tensor_shape[0], kv_tensor.get_shape()[2],
                                                 state_tensor_shape[2], state_tensor_shape[3]});
                           state_tensor = kv_tensor;
                           state_tensor_shape = state_tensor.get_shape();
                        }
                        ov::Coordinate begin = {0, 0, 0, 0};
                        ov::Coordinate end = {state_tensor_shape[0], static_cast<uint32_t>(pos_data[0]),
                                              state_tensor_shape[2], state_tensor_shape[3]};
                        ov::Tensor new_state_tensor(state_tensor, begin, end);
                        state.set_state(new_state_tensor);
                    }
                    r_ctx->stateful_kv_size = pos_data[0] + 1;
                }
            }

            decoder_end_time = ggml_time_us();
            conversion_end_time = decoder_end_time;
            compile_end_time = decoder_end_time;
        } else {
            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache.erase(key);
            }

            std::shared_ptr<ov::Model> model;
            auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

            ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static, stateful);
            decoder_end_time = ggml_time_us();

            auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
            model = ov::frontend::ggml::FrontEnd::convert(input_model);
            ggml_decoder->clear_model_weights();
            conversion_end_time = ggml_time_us();

            if (getenv("GGML_OPENVINO_DUMP_IR")) {
                char timestamped_filename[64];
                auto timestamp = (long long) ggml_time_us();
                snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%lld.xml", timestamp);
                ov::serialize(model, timestamped_filename);
            }

            ov::CompiledModel compiled_model;
            auto remote_context = ggml_openvino_get_remote_context();
            if (remote_context.has_value()) {
                compiled_model = core.compile_model(model, remote_context.value(), config);
            } else {
                compiled_model = core.compile_model(model, device, config);
            }
            compile_end_time = ggml_time_us();
            infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
            entry->ptr = ggml_decoder;

            std::vector<std::string> ov_input_names;
            std::vector<std::string> ov_output_names;
            for (const auto & ov_param : model->get_parameters()) {
                ov_input_names.push_back(ov_param->get_friendly_name());
            }
            for (const auto & ov_output : model->get_results()) {
                ov_output_names.push_back(ov_output->get_friendly_name());
            }

            {
                std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
                r_ctx->infer_request_cache[key] = infer_request;
                r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
                r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
            }

            if (stateful) {
                const auto * inp_pos = get_inp_pos_tensor(cgraph);
                auto pos_shape = ggml_decoder->get_shape(inp_pos);
                r_ctx->stateful_kv_size = pos_shape[3];
                const auto kv_param_res_names = ggml_decoder->get_kv_param_res_names();
                for (const auto& pair : kv_param_res_names) {
                    r_ctx->kv_state_input_name_map[pair.first+pair.second] = pair.first;
                }
            }
        }

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            ov_input_names = r_ctx->ov_input_names_cache[key];
            ov_output_names = r_ctx->ov_output_names_cache[key];
        }

        for (size_t i = 0; i < ov_input_names.size(); i++) {
            auto param_name = ov_input_names[i];
            auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        struct OutputCopy {
            ov::Tensor tensor;
            ggml_tensor * dst;
            bool copy_back = false;
        };
        std::vector<OutputCopy> output_copies;
        output_copies.reserve(ov_output_names.size());

        for (size_t i = 0; i < ov_output_names.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names[i]);
            auto output_shape = infer_request->get_output_tensor(i).get_shape();
            auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
            const bool direct_output = can_bind_ov_output_directly(output_shape, ggml_tensor);
            auto output_tensor = direct_output ?
                ov::Tensor(output_type, output_shape, ggml_tensor->data) :
                ov::Tensor(output_type, output_shape);
            infer_request->set_output_tensor(i, output_tensor);
            output_copies.push_back({output_tensor, ggml_tensor, !direct_output});
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        for (const auto & output_copy : output_copies) {
            if (output_copy.copy_back) {
                copy_ov_output_to_ggml(output_copy.tensor, output_copy.dst);
            }
        }

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names[i], output_tensor, output_tensor.data());
            }
        }

        if (getenv("GGML_OPENVINO_PROFILING")) {
            GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
            GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
            if (!cache_hit) {
                GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
                GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
            }
            GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
        }
    }

    return GGML_STATUS_SUCCESS;
}

enum ggml_status ov_graph_compute_static(ggml_cgraph * cgraph, std::shared_ptr<ov_runtime_context> r_ctx) {
    auto & core = ov_singleton_core();

    auto get_prefill_chunk_size = [] {
        const char * chunk_size_str = getenv("GGML_OPENVINO_PREFILL_CHUNK_SIZE");
        if (chunk_size_str && atoi(chunk_size_str) > 0) {
            return atoi(chunk_size_str);
        }
        return 256;
    };

    static std::string device = "NPU";
    static auto is_static = true;
    static auto stateful = false;
    static auto prefill_chunk_size = get_prefill_chunk_size();
    const auto & config = ggml_openvino_get_compile_config();

    if (is_naive(cgraph)) {
        return naive_compute(cgraph, core, device, config);
    }

    auto start_time = ggml_time_us();

    std::shared_ptr<GgmlOvDecoder> ggml_decoder;
    std::shared_ptr<ov::InferRequest> infer_request;
    ModelParams m_params;
    ComputeParams c_params;
    std::tie(m_params, c_params) = GgmlOvDecoder::compute_llm_params(cgraph, is_static);

    const auto * inp_pos = get_inp_pos_tensor(cgraph);
    const auto is_prefill = c_params.input_len > 1;
    graph_key key(cgraph);
    bool cache_hit;

    int64_t decoder_end_time;
    int64_t conversion_end_time;
    int64_t compile_end_time;
    int64_t infer_end_time;

    std::shared_ptr<decoder_runtime_ctx> entry;
    ModelParams old_m_params;

    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        auto it = r_ctx->decoder_cache.find(key);
        cache_hit = it != r_ctx->decoder_cache.end();
        if (cache_hit) {
            entry = it->second;
        } else {
            auto mutex = std::make_shared<std::mutex>();
            entry = std::make_shared<decoder_runtime_ctx>(mutex);
            r_ctx->decoder_cache[key] = entry;
        }
    }

    std::lock_guard<std::mutex> lock(*(entry->mutex));

    if (cache_hit) {
        ggml_decoder = entry->ptr;
        old_m_params = ggml_decoder->get_model_params();
        cache_hit = old_m_params.can_reuse_statically(m_params);
    }

    if (cache_hit) {
        std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
        ggml_decoder->m_is_prefill = is_prefill;
        ggml_decoder->set_model_params(m_params);
        ggml_decoder->set_compute_params(c_params);
        if (old_m_params.kv_buffer_changed(m_params)) {
            ggml_decoder->update_io(cgraph);
        }
        ggml_decoder->add_extra_inputs();
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            infer_request =
                is_prefill ? r_ctx->infer_request_cache_prefill.at(key) : r_ctx->infer_request_cache.at(key);
        }

        decoder_end_time = ggml_time_us();
        conversion_end_time = decoder_end_time;
        compile_end_time = decoder_end_time;
    } else {
        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            r_ctx->infer_request_cache.erase(key);
            r_ctx->infer_request_cache_prefill.erase(key);
        }

        std::shared_ptr<ov::Model> model;
        auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph);

        ggml_decoder = std::make_shared<GgmlOvDecoder>(cgraph, m_params, c_params, model_weights, is_static, stateful,
                                                       is_prefill, prefill_chunk_size);
        decoder_end_time = ggml_time_us();

        auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(ggml_decoder);
        model = ov::frontend::ggml::FrontEnd::convert(input_model);
        ggml_decoder->clear_model_weights();
        conversion_end_time = ggml_time_us();

        if (getenv("GGML_OPENVINO_DUMP_IR")) {
            char timestamped_filename[64];
            auto timestamp = (long long) ggml_time_us();
            snprintf(timestamped_filename, sizeof(timestamped_filename), "model_%s_%lld.xml",
                     is_prefill ? "prefill" : "decode", timestamp);
            ov::serialize(model, timestamped_filename);
        }

        ov::CompiledModel compiled_model;
        auto remote_context = ggml_openvino_get_remote_context();
        if (remote_context.has_value()) {
            compiled_model = core.compile_model(model, remote_context.value(), config);
        } else {
            compiled_model = core.compile_model(model, device, config);
        }

        infer_request = std::make_shared<ov::InferRequest>(compiled_model.create_infer_request());
        compile_end_time = ggml_time_us();

        entry->ptr = ggml_decoder;

        std::vector<std::string> ov_input_names;
        std::vector<std::string> ov_output_names;
        for (const auto & ov_param : model->get_parameters()) {
            ov_input_names.push_back(ov_param->get_friendly_name());
        }
        for (const auto & ov_output : model->get_results()) {
            ov_output_names.push_back(ov_output->get_friendly_name());
        }

        {
            std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
            if (is_prefill) {
                r_ctx->infer_request_cache_prefill[key] = infer_request;
            } else {
                r_ctx->infer_request_cache[key] = infer_request;
            }
            r_ctx->ov_input_names_cache[key] = std::move(ov_input_names);
            r_ctx->ov_output_names_cache[key] = std::move(ov_output_names);
        }
    }

    std::vector<std::string> ov_input_names_local;
    std::vector<std::string> ov_output_names_local;
    {
        std::lock_guard<std::mutex> map_lock(r_ctx->ctx_mutex);
        ov_input_names_local = r_ctx->ov_input_names_cache[key];
        ov_output_names_local = r_ctx->ov_output_names_cache[key];
    }

    if (is_prefill) {
        auto inp_len = inp_pos->ne[0];
        for (int chunk_index = 0; chunk_index * prefill_chunk_size < inp_len; chunk_index++) {
            struct StaticOutputCopy {
                ov::Tensor tensor;
                ggml_tensor * dst;
                bool copy_back = false;
            };
            std::vector<StaticOutputCopy> output_copies;
            output_copies.reserve(ov_output_names_local.size());

            for (size_t i = 0; i < ov_input_names_local.size(); i++) {
                auto param_name = ov_input_names_local[i];
                auto input_tensor = get_ov_input_tensor_static_prefill(ggml_decoder, param_name, chunk_index);
                infer_request->set_input_tensor(i, input_tensor);

                if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                    const auto input_tensor = infer_request->get_input_tensor(i);
                    print_input_tensor_info(param_name, input_tensor);
                }
            }

            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
                auto output_shape = infer_request->get_output_tensor(i).get_shape();
                auto ggml_shape = GgmlOvDecoder::get_shape(ggml_tensor);
                auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
                const auto output_ne = ov_shape_size(output_shape);
                const auto ggml_ne = static_cast<size_t>(ggml_nelements(ggml_tensor));
                (void) ggml_shape;
                (void) output_ne;
                (void) ggml_ne;
                const bool skip_output_copy = is_view_like_output(ggml_tensor) && is_static_cache_view_output(ggml_tensor);
                const bool direct_output = can_bind_ov_output_directly(output_shape, ggml_tensor);
                auto output_tensor = direct_output ?
                    ov::Tensor(output_type, output_shape, ggml_tensor->data) :
                    ov::Tensor(output_type, output_shape);
                infer_request->set_output_tensor(i, output_tensor);
                output_copies.push_back({output_tensor, ggml_tensor, !direct_output && !skip_output_copy});
            }

            infer_request->infer();

            for (const auto & output_copy : output_copies) {
                if (output_copy.copy_back) {
                    copy_static_prefill_output_to_ggml(
                        output_copy.tensor, output_copy.dst, inp_len, prefill_chunk_size, chunk_index);
                }
            }

            if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
                for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                    const auto output_tensor = infer_request->get_output_tensor(i);
                    print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
                }
            }
        }
        infer_end_time = ggml_time_us();
    } else {
        for (size_t i = 0; i < ov_input_names_local.size(); i++) {
            auto param_name = ov_input_names_local[i];
            auto input_tensor = get_ov_input_tensor_static_decode(ggml_decoder, param_name);
            infer_request->set_input_tensor(i, input_tensor);

            if (getenv("GGML_OPENVINO_DEBUG_INPUT")) {
                const auto input_tensor = infer_request->get_input_tensor(i);
                print_input_tensor_info(param_name, input_tensor);
            }
        }

        struct StaticDecodeOutputCopy {
            ov::Tensor tensor;
            ggml_tensor * dst;
            bool copy_back = false;
        };
        std::vector<StaticDecodeOutputCopy> output_copies;
        output_copies.reserve(ov_output_names_local.size());

        for (size_t i = 0; i < ov_output_names_local.size(); i++) {
            auto * ggml_tensor = ggml_decoder->get_model_outputs().at(ov_output_names_local[i]);
            auto output_shape = infer_request->get_output_tensor(i).get_shape();
            auto output_type = ggml_decoder->get_ov_type(ggml_tensor);
            const bool skip_output_copy = is_view_like_output(ggml_tensor) && is_static_cache_view_output(ggml_tensor);
            const bool direct_output = can_bind_ov_output_directly(output_shape, ggml_tensor);
            auto output_tensor = direct_output ?
                ov::Tensor(output_type, output_shape, ggml_tensor->data) :
                ov::Tensor(output_type, output_shape);
            infer_request->set_output_tensor(i, output_tensor);
            output_copies.push_back({output_tensor, ggml_tensor, !direct_output && !skip_output_copy});
        }

        infer_request->infer();
        infer_end_time = ggml_time_us();

        for (const auto & output_copy : output_copies) {
            if (output_copy.copy_back) {
                copy_ov_output_to_ggml(output_copy.tensor, output_copy.dst);
            }
        }

        if (getenv("GGML_OPENVINO_DEBUG_OUTPUT")) {
            for (size_t i = 0; i < ov_output_names_local.size(); i++) {
                const auto output_tensor = infer_request->get_output_tensor(i);
                print_output_tensor_info(ov_output_names_local[i], output_tensor, output_tensor.data());
            }
        }
    }

    if (getenv("GGML_OPENVINO_PROFILING")) {
        GGML_LOG_INFO("\nGGML OpenVINO Backend: \n");
        GGML_LOG_INFO("  - Graph decoder time: %ld ms \n", (decoder_end_time - start_time) / 1000);
        if (!cache_hit) {
            GGML_LOG_INFO("  - Graph conversion time: %ld ms \n", (conversion_end_time - decoder_end_time) / 1000);
            GGML_LOG_INFO("  - Graph compile time: %ld ms \n", (compile_end_time - conversion_end_time) / 1000);
        }
        GGML_LOG_INFO("  - Graph inference time: %ld ms \n", (infer_end_time - compile_end_time) / 1000);
    }

    return GGML_STATUS_SUCCESS;
}

bool is_naive(ggml_cgraph * cgraph) {
    constexpr int naive_graph_size_threshold = 20;
    int count = 0;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i]->op != GGML_OP_NONE) {
            count++;
        }
    }
    return count < naive_graph_size_threshold;
}

enum ggml_status naive_compute(ggml_cgraph * cgraph,
                               ov::Core & core,
                               const std::string & device,
                               const ov::AnyMap & config) {
    if (cgraph->n_nodes == 1 && (cgraph->nodes[0]->op == GGML_OP_NONE || cgraph->nodes[0]->op == GGML_OP_VIEW)) {
        return GGML_STATUS_SUCCESS;
    }

    bool naive = true;
    auto model_weights = GgmlOvDecoder::create_weight_nodes(cgraph, naive);
    auto decoder = std::make_shared<GgmlOvDecoder>(cgraph, model_weights);
    auto input_model = std::make_shared<ov::frontend::ggml::InputModel>(decoder);
    auto model = ov::frontend::ggml::FrontEnd::convert(input_model, naive);
    if (getenv("GGML_OPENVINO_DUMP_IR")) {
        ov::serialize(model, "IR_naive.xml");
    }

    std::shared_ptr<ov::InferRequest> infer_request;
    auto remote_context = ggml_openvino_get_remote_context();
    if (cgraph->nodes[0]->op == GGML_OP_MUL_MAT) {
        // TODO ACCURACY hint triggers a bug in GPU plugin/driver on Lunar Lake. Remove once CVS-182166 is resolved
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::PERFORMANCE));
    } else {
        core.set_property(device, ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
    }
    if (remote_context.has_value()) {
        infer_request = std::make_shared<ov::InferRequest>(
            core.compile_model(model, remote_context.value(), config).create_infer_request());
    } else {
        infer_request =
            std::make_shared<ov::InferRequest>(core.compile_model(model, device, config).create_infer_request());
    }

    auto ov_params = model->get_parameters();
    for (size_t i = 0; i < ov_params.size(); i++) {
        auto param_name = ov_params[i]->get_friendly_name();
        auto input_tensor = get_ov_input_tensor(decoder, param_name);
        infer_request->set_input_tensor(i, input_tensor);
    }

    auto ov_results = model->get_results();
    struct NaiveOutputCopy {
        ov::Tensor tensor;
        ggml_tensor * dst;
        bool copy_back = false;
    };
    std::vector<NaiveOutputCopy> output_copies;
    output_copies.reserve(ov_results.size());

    for (size_t i = 0; i < ov_results.size(); i++) {
        auto * ggml_tensor = decoder->get_model_outputs().at(ov_results[i]->get_friendly_name());
        auto output_shape = infer_request->get_output_tensor(i).get_shape();
        auto output_type = decoder->get_ov_type(ggml_tensor);
        const bool direct_output = can_bind_ov_output_directly(output_shape, ggml_tensor);
        auto output_tensor = direct_output ?
            ov::Tensor(output_type, output_shape, ggml_tensor->data) :
            ov::Tensor(output_type, output_shape);
        infer_request->set_output_tensor(i, output_tensor);
        output_copies.push_back({output_tensor, ggml_tensor, !direct_output});
    }

    infer_request->infer();
    for (const auto & output_copy : output_copies) {
        if (output_copy.copy_back) {
            copy_ov_output_to_ggml(output_copy.tensor, output_copy.dst);
        }
    }
    return GGML_STATUS_SUCCESS;
}

namespace {
ov::Tensor convert_ggml_input_to_ov(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & name) {
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(name);

    if (ggml_tensor->extra != nullptr) {
        // GGML_LOG_DEBUG("Using ggml_tensor->extra as ov::Tensor for input: %s\n", name.c_str());
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(ggml_tensor->extra);
        if (extra_base->type == ggml_openvino_extra_base::Type::TENSOR) {
            auto * tensor_extra = static_cast<ggml_openvino_tensor_extra *>(extra_base);
            return *tensor_extra->tensor;
        }
    }

    // GGML_LOG_DEBUG("Converting ggml tensor to ov::Tensor for input: %s\n", name.c_str());
    auto * input_data = ggml_tensor->data;
    ov::Shape input_shape = ggml_decoder->get_shape(ggml_tensor);
    auto input_tensor = ov::Tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape, input_data);
    return input_tensor;
}
}  // namespace

ov::Tensor get_ov_input_tensor(std::shared_ptr<GgmlOvDecoder> ggml_decoder, const std::string & param_name) {
    ov::Tensor input_tensor;
    if (ggml_decoder->get_model_extra_inputs().find(param_name) != ggml_decoder->get_model_extra_inputs().end()) {
        input_tensor = *ggml_decoder->get_model_extra_input_values().at(param_name);
    } else {
        input_tensor = convert_ggml_input_to_ov(ggml_decoder, param_name);
    }
    return input_tensor;
}

ov::Tensor get_ov_input_tensor_static_decode(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                             const std::string & param_name) {
    // NPU decoding stage
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        size_t lane_width = 1;
        if (GgmlOvDecoder::is_kv_idx(ggml_tensor, op) && ggml_decoder->get_input_len() > 0 &&
            ggml_tensor->ne[0] % ggml_decoder->get_input_len() == 0) {
            lane_width = ggml_tensor->ne[0] / ggml_decoder->get_input_len();
        }
        ov::Shape input_shape = {1, 1, 1, lane_width};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->type == GGML_TYPE_I32) {
            std::memcpy(input_tensor.data(), ggml_tensor->data, lane_width * sizeof(int32_t));
        } else if (ggml_tensor->type == GGML_TYPE_I64) {
            std::memcpy(input_tensor.data(), ggml_tensor->data, lane_width * sizeof(int64_t));
        } else {
            throw std::runtime_error("Unexpected tensor type for " + param_name);
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        ov::Shape input_shape = {1, 1, 1, 1};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        int32_t inp_out_id = *((int32_t *) ggml_tensor->data);
        assert(ggml_tensor->ne[0] == 1);
        assert(inp_out_id == 0);
        *input_tensor.data<int32_t>() = inp_out_id;
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data = pad_input<float>(ggml_tensor, 1, context_size, -INFINITY);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, 1, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + context_size, data_ptr);
        return input_tensor;
    }

    return get_ov_input_tensor(ggml_decoder, param_name);
}

ov::Tensor get_ov_input_tensor_static_prefill(std::shared_ptr<GgmlOvDecoder> ggml_decoder,
                                              const std::string & param_name,
                                              int chunk_index) {
    // NPU prompt processing stage
    const auto * ggml_tensor = ggml_decoder->get_input_ggml_tensor(param_name);
    const auto * op = ggml_decoder->get_tensor_used_op(ggml_tensor);

    const size_t input_len = ggml_decoder->get_input_len();
    const size_t chunk_size = ggml_decoder->m_prefill_chunk_size;
    const size_t chunk_valid_size = std::min(chunk_size, input_len - chunk_index * chunk_size);

    if (GgmlOvDecoder::is_inp_tok(ggml_tensor, op) || GgmlOvDecoder::is_inp_pos(ggml_tensor, op) ||
        GgmlOvDecoder::is_kv_idx(ggml_tensor, op)) {
        size_t lane_width = 1;
        if (GgmlOvDecoder::is_kv_idx(ggml_tensor, op) && input_len > 0 && ggml_tensor->ne[0] % input_len == 0) {
            lane_width = ggml_tensor->ne[0] / input_len;
        }
        const size_t input_chunk_size = chunk_size * lane_width;
        const size_t input_chunk_valid_size = chunk_valid_size * lane_width;
        const size_t input_chunk_pad_size = input_chunk_size - input_chunk_valid_size;
        ov::Shape input_shape = {1, 1, 1, input_chunk_size};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        // copy the chunk_index-th chunk from ggml_tensor
        size_t element_size = ggml_type_size(ggml_tensor->type);
        void * input_data = (char *) ggml_tensor->data + chunk_index * input_chunk_size * element_size;
        std::memcpy(input_tensor.data(), input_data, input_chunk_valid_size * element_size);
        // pad the rest with last_value + 1, so that kv's of padded positions are inserted
        // to the next row after the valids row in the kvcache
        if (input_chunk_pad_size > 0) {
            if (ggml_tensor->type == GGML_TYPE_I32) {
                int32_t last_value =
                    *((int32_t *) ggml_tensor->data + (chunk_index * input_chunk_size + input_chunk_valid_size - 1));
                int32_t * output_data = input_tensor.data<int32_t>();
                std::fill(output_data + input_chunk_valid_size, output_data + input_chunk_size, last_value + 1);
            } else if (ggml_tensor->type == GGML_TYPE_I64) {
                int64_t last_value =
                    *((int64_t *) ggml_tensor->data + (chunk_index * input_chunk_size + input_chunk_valid_size - 1));
                int64_t * output_data = input_tensor.data<int64_t>();
                std::fill(output_data + input_chunk_valid_size, output_data + input_chunk_size, last_value + 1);
            } else {
                throw std::runtime_error("Unexpected tensor type for " + param_name);
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_output_idx(ggml_tensor, op)) {
        size_t output_len = ggml_decoder->get_compute_params().output_len;
        ov::Shape input_shape = {1, 1, 1, output_len};
        ov::Tensor input_tensor(ggml_decoder->get_ov_type(ggml_tensor), input_shape);
        if (ggml_tensor->ne[0] == 0) {
            *input_tensor.data<int32_t>() = 0;
        } else {
            auto * data_addr = input_tensor.data<int32_t>();
            for (size_t i = 0; i < output_len; i++) {
                data_addr[i] = ((int32_t *) ggml_tensor->data)[i] % chunk_size;
            }
        }
        return input_tensor;
    }

    if (GgmlOvDecoder::is_inp_mask(ggml_tensor, op)) {
        size_t cols = ggml_tensor->ne[0];
        size_t rows = ggml_tensor->ne[1];
        float * ggml_data = (float *) ggml_tensor->data + chunk_index * chunk_size * cols;
        size_t chunk_valid_rows = std::min(chunk_size, rows - chunk_index * chunk_size);
        size_t context_size = ggml_decoder->get_ctx_size();
        std::vector<float> padded_data =
            pad_input<float>(ggml_data, chunk_valid_rows, cols, chunk_size, context_size, -INFINITY);
        set_zero_diagonal(padded_data, chunk_size, context_size);
        ov::Tensor input_tensor(ov::element::f32, ov::Shape{1, 1, chunk_size, context_size});
        auto * data_ptr = input_tensor.data<float>();
        std::copy(padded_data.begin(), padded_data.begin() + chunk_size * context_size, data_ptr);
        return input_tensor;
    }

    auto input_tensor = get_ov_input_tensor(ggml_decoder, param_name);
    auto input_shape = input_tensor.get_shape();
    auto padded_shape = input_shape;
    auto axis_it = std::find(padded_shape.begin(), padded_shape.end(), input_len);
    if (axis_it == padded_shape.end()) {
        return input_tensor;
    }

    const size_t axis = std::distance(padded_shape.begin(), axis_it);
    padded_shape[axis] = chunk_size;
    ov::Tensor padded_tensor(input_tensor.get_element_type(), padded_shape);
    std::memset(padded_tensor.data(), 0, padded_tensor.get_byte_size());

    size_t outer_size = 1;
    for (size_t i = 0; i < axis; ++i) {
        outer_size *= input_shape[i];
    }
    size_t inner_size = 1;
    for (size_t i = axis + 1; i < input_shape.size(); ++i) {
        inner_size *= input_shape[i];
    }

    const size_t element_size = input_tensor.get_byte_size() / input_tensor.get_size();
    const size_t valid_len = std::min(chunk_size, input_len - chunk_index * chunk_size);
    const size_t src_axis_offset = chunk_index * chunk_size;
    auto * src = static_cast<const uint8_t *>(input_tensor.data());
    auto * dst = static_cast<uint8_t *>(padded_tensor.data());
    for (size_t outer = 0; outer < outer_size; ++outer) {
        const size_t src_offset = (outer * input_len + src_axis_offset) * inner_size * element_size;
        const size_t dst_offset = outer * chunk_size * inner_size * element_size;
        std::memcpy(dst + dst_offset, src + src_offset, valid_len * inner_size * element_size);
    }

    return padded_tensor;
}

size_t checksum(const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    size_t sum = 0;
    for (size_t i = 0; i < size; ++i) {
        sum += (uint8_t) i;
        sum += bytes[i];
    }
    return sum;
}

void print_input_tensor_info(const std::string & name, const ov::Tensor & tensor) {
    std::cout << "Input name: " << name << ", Input shape: " << tensor.get_shape() << ", Address: " << tensor.data()
              << std::endl;
    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        if (name.find("self_kq_mask") == std::string::npos) {
            std::cout << *(tensor.data<float>()) << std::endl;
        } else {
            size_t rows = tensor.get_shape()[2];
            size_t cols = tensor.get_shape()[3];
            auto * data = tensor.data<float>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    float val = data[i * cols + j];
                    if (std::isinf(val) && val < 0) {
                        std::cout << std::setw(5) << "-inf";
                    } else {
                        std::cout << std::setw(5) << val;
                    }
                }
                std::cout << std::endl;
            }
        }

        break;
    }
    case ov::element::f16:
        std::cout << *(tensor.data<ov::float16>()) << std::endl;
        break;
    case ov::element::i32:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int32_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    case ov::element::i64:
        for (size_t i = 0; i < tensor.get_size(); ++i) {
            std::cout << tensor.data<int64_t>()[i] << " ";
        }
        std::cout << std::endl;
        break;
    default:
        break;
    }
}

void print_output_tensor_info(const std::string & name, const ov::Tensor & tensor, const void * output_dst) {
    std::cout << "Output name: " << name << ", Output shape: " << tensor.get_shape() << ", Address: " << output_dst
              << std::endl;

    auto print_float_stats = [](const std::string & type_name, size_t size, auto get_value) {
        if (size == 0) {
            return;
        }

        float first = get_value(0);
        float min = first;
        float max = first;
        double sum = first;

        for (size_t i = 1; i < size; ++i) {
            float v = get_value(i);
            if (v < min) {
                min = v;
            }
            if (v > max) {
                max = v;
            }
            sum += v;
        }
        double mean = sum / size;

        std::cout << std::right << std::setw(6) << type_name << std::right << std::setw(12) << "First" << std::setw(12)
                  << "Min" << std::setw(12) << "Max" << std::setw(12) << "Mean" << std::endl;
        std::cout << std::right << std::setw(6) << "" << std::right << std::setw(12) << first << std::setw(12) << min
                  << std::setw(12) << max << std::setw(12) << mean << std::endl;
    };

    switch (tensor.get_element_type()) {
    case ov::element::f32: {
        const float * data = tensor.data<float>();
        size_t size = tensor.get_size();
        print_float_stats("[f32]", size, [data](size_t i) { return data[i]; });
        break;
    }
    case ov::element::f16: {
        const ov::float16 * data = tensor.data<ov::float16>();
        size_t size = tensor.get_size();
        print_float_stats("[f16]", size, [data](size_t i) { return static_cast<float>(data[i]); });
        break;
    }
    default:
        break;
    }
}

void set_zero_diagonal(std::vector<float> & matrix, size_t rows, size_t cols) {
    for (size_t i = 0; i < rows; ++i) {
        size_t diag_col = std::min(i, cols - 1);
        matrix[i * cols + diag_col] = 0.0f;
    }
}

const ggml_tensor * get_inp_pos_tensor(ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; ++i) {
        auto * op = cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            auto * src = op->src[j];
            if (src == nullptr) {
                break;
            }
            if (GgmlOvDecoder::is_inp_pos(src, op)) {
                return src;
            }
        }
    }
    GGML_LOG_ERROR("get_inp_pos_tensor: inp_pos not found in cgraph");
    throw std::runtime_error("get_inp_pos_tensor: inp_pos not found in cgraph");
}

bool get_is_prefill(const ggml_tensor * inp_pos) {
    return inp_pos->ne[0] > 1;
}

#pragma GCC diagnostic pop
