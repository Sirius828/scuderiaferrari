#include "track_perception_cpp/shm_reader.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>

namespace track_perception_cpp {

ShmReader::ShmReader(std::string name) : name_(std::move(name)) {}

ShmReader::~ShmReader() { close(); }

bool ShmReader::connect() {
  if (data_ != nullptr) {
    return true;
  }

  std::string path = "/dev/shm/" + name_;
  fd_ = ::open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    return false;
  }

  struct stat st {};
  if (::fstat(fd_, &st) != 0 || st.st_size <= static_cast<off_t>(kHeaderSize)) {
    close();
    return false;
  }

  mapped_size_ = static_cast<size_t>(st.st_size);
  data_ = static_cast<uint8_t*>(::mmap(nullptr, mapped_size_, PROT_READ, MAP_SHARED, fd_, 0));
  if (data_ == MAP_FAILED) {
    data_ = nullptr;
    close();
    return false;
  }
  return true;
}

bool ShmReader::readLatest(Frame& frame) {
  if (data_ == nullptr && !connect()) {
    return false;
  }

  uint64_t fid = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  std::memcpy(&fid, data_, sizeof(fid));
  std::memcpy(&width, data_ + 8, sizeof(width));
  std::memcpy(&height, data_ + 12, sizeof(height));

  if (fid == 0 || fid == last_fid_ || width == 0 || height == 0) {
    return false;
  }

  size_t image_size = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
  if (kHeaderSize + image_size > mapped_size_) {
    return false;
  }

  frame.fid = fid;
  frame.width = width;
  frame.height = height;
  frame.image.create(static_cast<int>(height), static_cast<int>(width), CV_8UC3);
  std::memcpy(frame.image.data, data_ + kHeaderSize, image_size);
  last_fid_ = fid;
  return true;
}

void ShmReader::close() {
  if (data_ != nullptr) {
    ::munmap(data_, mapped_size_);
    data_ = nullptr;
  }
  if (fd_ >= 0) {
    ::close(fd_);
    fd_ = -1;
  }
  mapped_size_ = 0;
}

}  // namespace track_perception_cpp
