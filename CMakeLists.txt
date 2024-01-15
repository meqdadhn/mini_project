cmake_minimum_required(VERSION 3.12)
project(TwoViewReconstruction)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)

# Find OpenCV 3.4
find_package(OpenCV 3.4 REQUIRED COMPONENTS core highgui imgproc features2d xfeatures2d)

# Set include directories
include_directories(include)

# Add your source files
add_executable(TwoViewReconstruction src/main.cpp src/TwoViewReconstruction.cpp)

# Link against OpenCV 3.4
target_link_libraries(TwoViewReconstruction ${OpenCV_LIBS})
