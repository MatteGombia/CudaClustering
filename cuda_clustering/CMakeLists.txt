cmake_minimum_required(VERSION 3.8)
project(cuda_clustering)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(CUDA  REQUIRED)
find_package(visualization_msgs REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

# Find include directories
include_directories(include ${catkin_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(${Eigen_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(/opt/ros/humble/include/pcl_ros)
include_directories(/opt/ros/humble/include/pcl_conversions)
include_directories(/opt/ros/humble/include/message_filters) #
include_directories(/opt/ros/humble/include/pcl_msgs)   
include_directories(/opt/ros/humble/include/tf2)   
include_directories(/opt/ros/humble/include/tf2_ros)   
include_directories(/opt/ros/humble/include/tf2_msgs)  
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_definitions(-std=c++17 -g -lboost_thread-mt)

add_library(${PROJECT_NAME} src/${PROJECT_NAME}.cpp)

add_executable(cuda_clustering_node src/cuda_clustering_node.cpp)

add_library(libcudacluster SHARED IMPORTED)
set_target_properties( libcudacluster PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libcudacluster.so )

add_library(libcudafilter SHARED IMPORTED)
set_target_properties( libcudafilter PROPERTIES IMPORTED_LOCATION ${CMAKE_SOURCE_DIR}/lib/libcudafilter.so )

link_directories(libcudacluster)
link_directories(libcudafilter)

target_link_libraries(${PROJECT_NAME}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  libcudacluster
  libcudafilter
)
ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  sensor_msgs
  visualization_msgs
)

target_link_libraries(${PROJECT_NAME}_node 
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
  libcudacluster
  libcudafilter
)

target_include_directories(cuda_clustering PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(cuda_clustering PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17

install(TARGETS cuda_clustering_node
  DESTINATION lib/${PROJECT_NAME})


# Install launch directory
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)
install(DIRECTORY
  lib
  DESTINATION share/${PROJECT_NAME}
)

# Install config directory
install(DIRECTORY
  config
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
