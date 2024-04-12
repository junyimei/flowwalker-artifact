cmake_minimum_required(VERSION 3.15)

project(flowwalker VERSION 1.0 LANGUAGES CXX CUDA)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(CUDA_PATH "/usr/local/cuda")
set(HEAD_DIR ${PROJECT_SOURCE_DIR}/include)

set(TEST_SAMPLER TestSampler.out)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/src/main.cu ${PROJECT_SOURCE_DIR}/src/util.cu ${PROJECT_SOURCE_DIR}/src/walk.cu)

add_executable(${TEST_SAMPLER} ${PROJECT_SOURCE_DIR}/src/sampler_test.cu ${PROJECT_SOURCE_DIR}/src/util.cu)

include_directories(${HEAD_DIR})

# set gpu architecture
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES "70;75;80")
set_property(TARGET ${TEST_SAMPLER} PROPERTY CUDA_ARCHITECTURES "70;75;80")

# Link gflags
target_link_libraries(${PROJECT_NAME} -lgflags)
target_link_libraries(${TEST_SAMPLER} -lgflags)


# enable compute-sanitizer
target_compile_options(${PROJECT_NAME} PUBLIC -lineinfo)
target_compile_options(${TEST_SAMPLER} PUBLIC -lineinfo)

target_compile_options(${PROJECT_NAME} PUBLIC -O3)
target_compile_options(${TEST_SAMPLER} PUBLIC -O3)

target_compile_options(${PROJECT_NAME} PUBLIC -maxrregcount=32)
target_compile_options(${TEST_SAMPLER} PUBLIC -maxrregcount=32)


# Enable dynamic parallelism
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)
target_link_libraries(${PROJECT_NAME} -lcudadevrt)

set_target_properties(${TEST_SAMPLER} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_compile_options(${TEST_SAMPLER} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --relocatable-device-code=true
        >)
target_link_libraries(${TEST_SAMPLER} -lcudadevrt)

# Link curand
target_link_libraries(${PROJECT_NAME} -lcurand)
target_link_libraries(${TEST_SAMPLER} -lcurand)
