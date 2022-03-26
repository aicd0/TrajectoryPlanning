cmake_minimum_required(VERSION 3.0.2)
project(robot_sim)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  std_msgs
  message_generation
)

find_package(gazebo REQUIRED)

# Generate service headers
add_service_files(
  FILES
  StepWorldSrvInfo.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_sim
   CATKIN_DEPENDS gazebo_ros roscpp std_msgs message_runtime
#  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${GAZEBO_INCLUDE_DIRS}
  "/opt/boost"
)

link_directories(
  ${GAZEBO_LIBRARY_DIRS}
)

# Generate plugins
add_library(step_world SHARED src/step_world.cpp)
target_link_libraries(step_world ${GAZEBO_LIBRARIES})
add_dependencies(step_world ${PROJECT_NAME}_gencpp)
