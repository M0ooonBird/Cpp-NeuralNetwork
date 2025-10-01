cmake_minimum_required(VERSION 3.18) 
project(MyNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


file(GLOB_RECURSE SOURCES
    "src/*.cpp"    # C++ 源文件
)
# 自动扫描头文件目录
file(GLOB_RECURSE HEADER_FILES "include/*.h" "include/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADER_FILES})
# 分组显示头文件
source_group("Include" FILES ${HEADER_FILES})


target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_sources(${PROJECT_NAME} PRIVATE
 "${CMAKE_CURRENT_SOURCE_DIR}/NatvisFile.natvis"
)