#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "track_perception_cpp/types.hpp"

namespace track_perception_cpp {

class ShmReader {
 public:
  explicit ShmReader(std::string name);
  ~ShmReader();

  bool connect();
  bool readLatest(Frame& frame);
  void close();

 private:
  static constexpr size_t kHeaderSize = 16;

  std::string name_;
  int fd_{-1};
  size_t mapped_size_{0};
  uint8_t* data_{nullptr};
  uint64_t last_fid_{0};
};

}  // namespace track_perception_cpp
