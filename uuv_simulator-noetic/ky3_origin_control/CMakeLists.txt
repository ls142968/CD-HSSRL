cmake_minimum_required(VERSION 3.0.2)
project(ky3_origin_control)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
  geometry_msgs
  uuv_gazebo_ros_plugins_msgs
  joy
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES ky3_origin_control
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

include_directories(
  src/robot_control.py
)
