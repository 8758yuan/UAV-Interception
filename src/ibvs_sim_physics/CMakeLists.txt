cmake_minimum_required(VERSION 3.16)
project(ibvs_sim_physics)

find_package(ament_cmake REQUIRED)
find_package(gz-sim8 REQUIRED)
find_package(gz-transport13 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)

add_library(paper_target_system SHARED src/paper_target_system.cc)
target_compile_features(paper_target_system PRIVATE cxx_std_17)
target_compile_options(paper_target_system PRIVATE -Wall -Wextra)
target_link_libraries(paper_target_system PRIVATE
  gz-sim8::gz-sim8 gz-transport13::gz-transport13 gz-plugin2::register)
install(TARGETS paper_target_system LIBRARY DESTINATION lib)

ament_package()
