cmake_minimum_required(VERSION 3.8)
project(openmv_cam)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EventTrackerUpdate.msg"
  DEPENDENCIES std_msgs
)

# rosidl installs the generated ``openmv_cam.msg`` Python package. Installing
# this package's pure-Python sources into the same destination explicitly
# avoids the duplicate ament_cmake_python target created when both macros use
# the project name.
ament_get_python_install_dir(PYTHON_INSTALL_DIR)
file(GLOB PACKAGE_PYTHON_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/openmv_cam/*.py")
install(FILES ${PACKAGE_PYTHON_SOURCES}
  DESTINATION ${PYTHON_INSTALL_DIR}/${PROJECT_NAME})
install(PROGRAMS scripts/openmv_cam_node DESTINATION lib/${PROJECT_NAME})
install(FILES launch/openmv.launch.py DESTINATION share/${PROJECT_NAME}/launch)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(openmv_cam_pytest test)
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
