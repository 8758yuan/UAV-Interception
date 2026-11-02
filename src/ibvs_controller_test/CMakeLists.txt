cmake_minimum_required(VERSION 3.16)
project(ibvs_controller_test)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(controller_truth_system SHARED src/controller_truth_system.cc)
target_compile_features(controller_truth_system PRIVATE cxx_std_17)
target_compile_options(controller_truth_system PRIVATE -Wall -Wextra)
target_link_libraries(controller_truth_system PRIVATE
  gz-sim8::gz-sim8 gz-transport13::gz-transport13 gz-plugin2::register)
install(TARGETS controller_truth_system LIBRARY DESTINATION lib)

ament_package()
