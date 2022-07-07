cmake_minimum_required(VERSION 3.1)
project(example_project)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x")



message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")




#add_executable
#add_executable(opencv_example src/ransac.cpp)
include_directories(src)
add_executable(linelidar  main.cpp src/line.cpp)

target_include_directories(linelidar PUBLIC "${EIGEN3_INCLUDE_DIR}")
#target_link_libraries
target_link_libraries(linelidar PRIVATE ${OpenCV_LIBS} )
#target_link_libraries(ran PRIVATE ${OpenCV_LIBS} )
#target_link_libraries(opencv_example PUBLIC ${EIGEN3_LIBRARIES})