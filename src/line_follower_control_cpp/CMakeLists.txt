cmake_minimum_required(VERSION 3.16)
project(line_follower_control_cpp)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(controller_node
  src/controller_node.cpp
)

target_include_directories(controller_node PRIVATE include)

ament_target_dependencies(controller_node
  rclcpp
  rcl_interfaces
  std_msgs
  geometry_msgs
  std_srvs
)

install(TARGETS controller_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_control_core
    test/test_control_core.cpp
    src/control_core.cpp
  )
  target_include_directories(test_control_core PRIVATE include)

  ament_add_gtest(test_finish_turn_state
    test/test_finish_turn_state.cpp
  )
  target_include_directories(test_finish_turn_state PRIVATE include)

  ament_add_gtest(test_guideboard_reverse_state
    test/test_guideboard_reverse_state.cpp
  )
  target_include_directories(test_guideboard_reverse_state PRIVATE include)
endif()

ament_package()
