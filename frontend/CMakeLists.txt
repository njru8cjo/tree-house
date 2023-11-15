project(frontend)

include_directories(${PROJECT_SOURCE_DIR})
add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME})