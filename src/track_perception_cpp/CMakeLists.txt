cmake_minimum_required(VERSION 3.16)
project(track_perception_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  third_party/rknn/include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(fused_perception_node
  src/fused_perception_node.cpp
  src/lane_decision.cpp
  src/rknn_model.cpp
  src/shm_reader.cpp
  src/yolo_detector.cpp
  src/yolo_seg.cpp
)

ament_target_dependencies(fused_perception_node
  ament_index_cpp
  rclcpp
  std_msgs
)

target_link_libraries(fused_perception_node
  ${OpenCV_LIBS}
  rknnrt
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

ament_package()
