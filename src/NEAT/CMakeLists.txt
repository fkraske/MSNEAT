project(NEAT)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_library(${PROJECT_NAME} STATIC main.cpp Random.cpp  "InnovationManager.h")
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})