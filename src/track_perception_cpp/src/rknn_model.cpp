#include "track_perception_cpp/rknn_model.hpp"

#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace track_perception_cpp {

namespace {

std::string dimsToString(const rknn_tensor_attr& attr) {
  std::ostringstream ss;
  ss << "[";
  for (uint32_t i = 0; i < attr.n_dims; ++i) {
    if (i > 0) {
      ss << ",";
    }
    ss << attr.dims[i];
  }
  ss << "]";
  return ss.str();
}

}  // namespace

RknnModel::~RknnModel() { release(); }

std::vector<uint8_t> RknnModel::readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("failed to open RKNN model: " + path);
  }
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  std::vector<uint8_t> data(static_cast<size_t>(size));
  if (!file.read(reinterpret_cast<char*>(data.data()), size)) {
    throw std::runtime_error("failed to read RKNN model: " + path);
  }
  return data;
}

rknn_core_mask RknnModel::coreMask(int core_id) {
  switch (core_id) {
    case 0:
      return RKNN_NPU_CORE_0;
    case 1:
      return RKNN_NPU_CORE_1;
    case 2:
      return RKNN_NPU_CORE_2;
    case -1:
      return RKNN_NPU_CORE_0_1_2;
    default:
      return RKNN_NPU_CORE_AUTO;
  }
}

bool RknnModel::load(const std::string& model_path, int core_id) {
  release();

  auto model_data = readFile(model_path);
  int ret = rknn_init(&ctx_, model_data.data(), static_cast<uint32_t>(model_data.size()), 0, nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_init failed for " << model_path << ", ret=" << ret << std::endl;
    return false;
  }

  ret = rknn_set_core_mask(ctx_, coreMask(core_id));
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_set_core_mask failed, core=" << core_id << ", ret=" << ret << std::endl;
    return false;
  }

  std::memset(&io_num_, 0, sizeof(io_num_));
  ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_query IN_OUT_NUM failed, ret=" << ret << std::endl;
    return false;
  }

  input_attrs_.resize(io_num_.n_input);
  output_attrs_.resize(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_input; ++i) {
    std::memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
    input_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &input_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_query INPUT_ATTR failed, ret=" << ret << std::endl;
      return false;
    }
  }
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    std::memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
    output_attrs_[i].index = i;
    ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &output_attrs_[i], sizeof(rknn_tensor_attr));
    if (ret != RKNN_SUCC) {
      std::cerr << "rknn_query OUTPUT_ATTR failed, ret=" << ret << std::endl;
      return false;
    }
  }

  const auto& input = input_attrs_[0];
  if (input.fmt == RKNN_TENSOR_NCHW) {
    input_channels_ = input.dims[1];
    input_height_ = input.dims[2];
    input_width_ = input.dims[3];
  } else {
    input_height_ = input.dims[1];
    input_width_ = input.dims[2];
    input_channels_ = input.dims[3];
  }

  std::cerr << "RKNN model loaded: " << model_path << " core=" << core_id
            << " inputs=" << io_num_.n_input << " outputs=" << io_num_.n_output << std::endl;
  for (uint32_t i = 0; i < io_num_.n_input; ++i) {
    const auto& attr = input_attrs_[i];
    std::cerr << "  input[" << i << "] name=" << attr.name << " dims=" << dimsToString(attr)
              << " n_elems=" << attr.n_elems << " size=" << attr.size
              << " fmt=" << attr.fmt << " type=" << attr.type << " qnt=" << attr.qnt_type
              << std::endl;
  }
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    const auto& attr = output_attrs_[i];
    std::cerr << "  output[" << i << "] name=" << attr.name << " dims=" << dimsToString(attr)
              << " n_elems=" << attr.n_elems << " size=" << attr.size
              << " fmt=" << attr.fmt << " type=" << attr.type << " qnt=" << attr.qnt_type
              << std::endl;
  }

  loaded_ = true;
  return true;
}

void RknnModel::release() {
  if (loaded_) {
    rknn_destroy(ctx_);
    ctx_ = 0;
    loaded_ = false;
  }
}

bool RknnModel::infer(const cv::Mat& input_nhwc_u8, std::vector<TensorData>& outputs, double* rknn_ms) {
  if (!loaded_) {
    return false;
  }
  if (!input_nhwc_u8.isContinuous()) {
    return false;
  }

  std::lock_guard<std::mutex> lock(mutex_);

  rknn_input input {};
  input.index = 0;
  input.type = RKNN_TENSOR_UINT8;
  input.fmt = RKNN_TENSOR_NHWC;
  input.size = static_cast<uint32_t>(input_nhwc_u8.total() * input_nhwc_u8.elemSize());
  input.buf = const_cast<uint8_t*>(input_nhwc_u8.ptr<uint8_t>());

  auto t0 = std::chrono::steady_clock::now();
  int ret = rknn_inputs_set(ctx_, io_num_.n_input, &input);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_inputs_set failed, ret=" << ret << std::endl;
    return false;
  }

  ret = rknn_run(ctx_, nullptr);
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_run failed, ret=" << ret << std::endl;
    return false;
  }

  std::vector<rknn_output> raw_outputs(io_num_.n_output);
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    raw_outputs[i].index = i;
    raw_outputs[i].want_float = 1;
    raw_outputs[i].is_prealloc = 0;
  }

  ret = rknn_outputs_get(ctx_, io_num_.n_output, raw_outputs.data(), nullptr);
  auto t1 = std::chrono::steady_clock::now();
  if (rknn_ms != nullptr) {
    *rknn_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  }
  if (ret != RKNN_SUCC) {
    std::cerr << "rknn_outputs_get failed, ret=" << ret << std::endl;
    return false;
  }

  outputs.clear();
  outputs.reserve(raw_outputs.size());
  for (uint32_t i = 0; i < io_num_.n_output; ++i) {
    TensorData tensor;
    const auto& attr = output_attrs_[i];
    for (uint32_t d = 0; d < attr.n_dims; ++d) {
      tensor.dims.push_back(attr.dims[d]);
    }
    size_t elems = attr.n_elems;
    tensor.data.resize(elems);
    std::memcpy(tensor.data.data(), raw_outputs[i].buf, elems * sizeof(float));
    outputs.push_back(std::move(tensor));
  }

  rknn_outputs_release(ctx_, io_num_.n_output, raw_outputs.data());
  return true;
}

}  // namespace track_perception_cpp
