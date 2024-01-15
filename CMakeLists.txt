cmake_minimum_required(VERSION 3.12)
project(TwoViewReconstruction)

find_package(OpenCV REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc features2d xfeatures2d)

add_executable(TwoViewReconstruction main.cpp)

target_link_libraries(TwoViewReconstruction ${OpenCV_LIBS})
