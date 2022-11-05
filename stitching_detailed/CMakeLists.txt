cmake_minimum_required(VERSION 2.6)

# set the project name and version
project (temp)

#set (CMAKE_CXX_STANDARD 11)
find_package(OpenCV REQUIRED) 

include_directories(${OpenCV_INCLUDE_DIRS}) # not needed for opencv>=4.0
add_executable(hello-opencv main.cpp)
target_link_libraries(hello-opencv ${OpenCV_LIBS})
