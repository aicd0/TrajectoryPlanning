cmake_minimum_required(VERSION 3.0.2)
project(robot_sim)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
  roscpp
  rospy
  message_generation
  std_msgs
  geometry_msgs
)

find_package(gazebo REQUIRED)

# Generate service headers
add_service_files(
  FILES
  PlaceTarget.srv
  StepWorld.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES robot_sim
   CATKIN_DEPENDS gazebo_ros roscpp rospy message_runtime std_msgs geometry_msgs
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

link_libraries(
  ${catkin_LIBRARIES}
  ${GAZEBO_LIBRARIES}
)

# Generate libraries
add_library(step_world SHARED src/step_world.cpp)
add_dependencies(step_world ${PROJECT_NAME}_gencpp)

# Generate executables
add_executable(place_target src/place_target.cpp)
add_dependencies(place_target ${PROJECT_NAME}_gencpp)