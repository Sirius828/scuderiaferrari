cmake_minimum_required(VERSION 3.16)
project(track_perception_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(RKNN_MODEL_ZOO_ROOT "/home/orangepi/Desktop/rknn_model_zoo" CACHE PATH "rknn_model_zoo root path")
set(PPOCR_DIRECTION_ROOT "/home/orangepi/Desktop/ppocr_direction_runtime/cpp" CACHE PATH "ppocr direction runtime cpp path")
set(TARGET_SOC "rk3588" CACHE STRING "Target Rockchip SoC")

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

add_subdirectory(${RKNN_MODEL_ZOO_ROOT}/3rdparty rknn_model_zoo_3rdparty.out)
add_subdirectory(${RKNN_MODEL_ZOO_ROOT}/utils rknn_model_zoo_utils.out)

include_directories(
  include
  third_party/rknn/include
  ${PPOCR_DIRECTION_ROOT}
  ${RKNN_MODEL_ZOO_ROOT}/utils
  ${LIBRKNNRT_INCLUDES}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(fused_perception_node
  src/fused_perception_node.cpp
  src/lane_decision.cpp
  src/rknn_model.cpp
  src/shm_reader.cpp
  src/yolo_detector.cpp
  src/yolo_seg.cpp
  ${PPOCR_DIRECTION_ROOT}/ppocr_direction_system.cc
  ${PPOCR_DIRECTION_ROOT}/direction_rule.cpp
  ${PPOCR_DIRECTION_ROOT}/postprocess.cc
  ${PPOCR_DIRECTION_ROOT}/clipper.cc
  ${PPOCR_DIRECTION_ROOT}/rknpu2/ppocr_system.cc
)

ament_target_dependencies(fused_perception_node
  ament_index_cpp
  rclcpp
  std_msgs
)

target_link_libraries(fused_perception_node
  fileutils
  imageutils
  imagedrawing
  ${OpenCV_LIBS}
  ${LIBRKNNRT}
)

install(TARGETS fused_perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY scripts
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
