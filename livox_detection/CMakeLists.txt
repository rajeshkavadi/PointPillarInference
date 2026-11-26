cmake_minimum_required(VERSION 3.0.2)
project(livox_detection)

find_package(catkin REQUIRED COMPONENTS
  rospy
  rosbag
  std_msgs
  sensor_msgs
  geometry_msgs
  visualization_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  tools/pointpillars_ros_cpu.py
  tools/pointpillars_cpu_inference.py
  tools/pointpillars_web_view.py
  tools/extract_bag_frame.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY configs launch rviz pointpillars checkpoints webapp
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
