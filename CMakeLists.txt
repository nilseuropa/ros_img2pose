cmake_minimum_required(VERSION 2.8.3)
project(ros_img2pose)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
)

catkin_package(
  CATKIN_DEPENDS
  rospy
)

catkin_python_setup()
