cmake_minimum_required(VERSION 3.16)
project(medical_softmax LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH
        /usr/local/Ascend/ascend-toolkit/latest)
endif()

if(NOT DEFINED SOC_VERSION)
    set(SOC_VERSION "Ascend910B3")
endif()

if(NOT DEFINED RUN_MODE)
    set(RUN_MODE "npu")
endif()

set(ASCEND_ARCH_PATH
    ${ASCEND_CANN_PACKAGE_PATH}/aarch64-linux)

include(
    ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake
)

ascendc_library(
    ascendc_kernels STATIC
    medical_softmax_kernel.cpp
    medical_fused_softmax_kernel.cpp
)

add_executable(
    medical_softmax_run
    main.cpp
)

target_link_libraries(
    medical_softmax_run PRIVATE
    ascendc_kernels ascendcl runtime
)

add_executable(
    medical_fused_softmax_run
    fused_main.cpp
)

target_link_libraries(
    medical_fused_softmax_run PRIVATE
    ascendc_kernels ascendcl runtime
)

add_library(
    medical_npu_runtime SHARED
    runtime_api.cpp
    fused_runtime_api.cpp
)

target_link_libraries(
    medical_npu_runtime PRIVATE
    ascendc_kernels ascendcl runtime pthread dl
)

foreach(target medical_softmax_run medical_fused_softmax_run medical_npu_runtime)
    target_include_directories(${target} PRIVATE
        ${ASCEND_CANN_PACKAGE_PATH}/include
        ${ASCEND_ARCH_PATH}/include
    )
    target_link_directories(${target} PRIVATE
        ${ASCEND_ARCH_PATH}/lib64
        ${ASCEND_CANN_PACKAGE_PATH}/lib64
    )
endforeach()
