cmake_minimum_required(VERSION 3.16)
project(track_perception_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(TARGET_SOC "rk3588" CACHE STRING "Target Rockchip SoC")

set(PPOCR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/third_party/ppocr)
set(PPOCR_UTILS_ROOT ${PPOCR_ROOT}/utils)

find_library(RKNNRT_LIB
  NAMES rknnrt
  PATHS /usr/lib /usr/lib/aarch64-linux-gnu
  NO_DEFAULT_PATH
  REQUIRED
)

find_package(ament_cmake REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

include_directories(
  include
  third_party/rknn/include
  ${PPOCR_ROOT}
  ${PPOCR_UTILS_ROOT}
  ${PPOCR_ROOT}/stb
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(fused_perception_node
  src/fused_perception_node.cpp
  src/guideboard_api_client.cpp
  src/guideboard_recognizer.cpp
  src/guideboard_route_policy.cpp
  src/lane_decision.cpp
  src/rknn_model.cpp
  src/shm_reader.cpp
  src/yolo_detector.cpp
  src/yolo_seg.cpp
  ${PPOCR_ROOT}/ppocr_direction_system.cc
  ${PPOCR_ROOT}/direction_rule.cpp
  ${PPOCR_ROOT}/postprocess.cc
  ${PPOCR_ROOT}/clipper.cc
  ${PPOCR_ROOT}/rknpu2/ppocr_system.cc
  ${PPOCR_UTILS_ROOT}/file_utils.c
  ${PPOCR_UTILS_ROOT}/image_utils.c
)

target_compile_definitions(fused_perception_node PRIVATE
  DISABLE_RGA
  DISABLE_LIBJPEG
)

ament_target_dependencies(fused_perception_node
  ament_index_cpp
  geometry_msgs
  rclcpp
  rcl_interfaces
  std_msgs
  std_srvs
)

target_link_libraries(fused_perception_node
  ${OpenCV_LIBS}
  ${RKNNRT_LIB}
  CURL::libcurl
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

install(DIRECTORY model
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_guideboard_api_client
    test/test_guideboard_api_client.cpp
    src/guideboard_api_client.cpp
    src/guideboard_recognizer.cpp
  )
  target_include_directories(test_guideboard_api_client PRIVATE include)
  target_link_libraries(test_guideboard_api_client CURL::libcurl)

  ament_add_gtest(test_guideboard_recognizer
    test/test_guideboard_recognizer.cpp
    src/guideboard_recognizer.cpp
  )
  target_include_directories(test_guideboard_recognizer PRIVATE include)

  ament_add_gtest(test_guideboard_route_policy
    test/test_guideboard_route_policy.cpp
    src/guideboard_route_policy.cpp
  )
  target_include_directories(test_guideboard_route_policy PRIVATE include)

  ament_add_gtest(test_lane_guideboard_wait
    test/test_lane_guideboard_wait.cpp
    src/lane_decision.cpp
  )
  target_include_directories(test_lane_guideboard_wait PRIVATE include)
  target_link_libraries(test_lane_guideboard_wait ${OpenCV_LIBS})

  ament_add_gtest(test_lane_decision_car_boundary
    test/test_lane_decision_car_boundary.cpp
    src/lane_decision.cpp
  )
  target_include_directories(test_lane_decision_car_boundary PRIVATE include)
  target_link_libraries(test_lane_decision_car_boundary ${OpenCV_LIBS})

  ament_add_gtest(test_lane_decision_coin_shadow
    test/test_lane_decision_coin_shadow.cpp
    src/lane_decision.cpp
  )
  target_include_directories(test_lane_decision_coin_shadow PRIVATE include)
  target_link_libraries(test_lane_decision_coin_shadow ${OpenCV_LIBS})

  ament_add_gtest(test_lane_decision_finish_stop
    test/test_lane_decision_finish_stop.cpp
    src/lane_decision.cpp
  )
  target_include_directories(test_lane_decision_finish_stop PRIVATE include)
  target_link_libraries(test_lane_decision_finish_stop ${OpenCV_LIBS})
endif()

ament_package()
