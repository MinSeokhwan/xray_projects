cmake_minimum_required(VERSION 3.6 FATAL_ERROR)

project(Simulation)

find_package(Geant4 REQUIRED ui_all vis_all)

include(${Geant4_USE_FILE})

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB include ${PROJECT_SOURCE_DIR}/include/*.hh)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_executable(NS src/sim.cc ${sources} ${include})
target_link_libraries(NS ${Geant4_LIBRARIES})

file(GLOB MACRO_FILES
  "mac/*.mac"
)
file(COPY ${MACRO_FILES} DESTINATION ${PROJECT_BINARY_DIR})

file(GLOB DATA_FILES
  "dat/*.dat"
)
file(COPY ${DATA_FILES} DESTINATION ${PROJECT_BINARY_DIR})

add_custom_target(Simulation DEPENDS NS)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)