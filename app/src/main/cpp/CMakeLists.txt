cmake_minimum_required(VERSION 3.22.1)
project(mcnpu)
add_library(mcnpu SHARED mcnpu.cpp bonsai_pq2.cpp ornith_kv.cpp ornith_runtime.cpp)
set(QNN_INC "" CACHE PATH "QNN headers")
set(LLAMA_CPP_SRC "" CACHE PATH "Optional llama.cpp source tree for real Ornith inference")
if(LLAMA_CPP_SRC)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
    set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
    set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
    set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
    set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LLAMA_CPP_SRC} ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)
    set(MCNPU_HAS_LLAMA 1)
else()
    set(MCNPU_HAS_LLAMA 0)
endif()
if(NOT QNN_INC)
    message(FATAL_ERROR "QNN_INC is required")
endif()
target_include_directories(mcnpu PRIVATE ${QNN_INC})
find_library(log-lib log)
target_link_libraries(mcnpu PRIVATE ${log-lib} dl)
target_compile_features(mcnpu PRIVATE cxx_std_17)
target_compile_options(mcnpu PRIVATE -fvisibility=hidden -ffunction-sections -fdata-sections -Wall -Wextra)
target_link_options(mcnpu PRIVATE -Wl,--gc-sections)

# Stamp the commit this binary was built from into the diagnostic output. The
# diag is read out of logs pasted into a chat, and more than one round has been
# spent reasoning about output that came from an APK older than the fix under
# discussion. Anything that cannot identify itself is not diagnosable.
set(MCNPU_BUILD_ID "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short=8 HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE _mcnpu_sha
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(_mcnpu_sha)
        set(MCNPU_BUILD_ID "${_mcnpu_sha}")
    endif()
endif()
message(STATUS "MCNPU_BUILD_ID=${MCNPU_BUILD_ID}")
target_compile_definitions(mcnpu PRIVATE MCNPU_BUILD_ID="${MCNPU_BUILD_ID}" MCNPU_HAS_LLAMA=${MCNPU_HAS_LLAMA})
if(LLAMA_CPP_SRC)
    target_include_directories(mcnpu PRIVATE ${LLAMA_CPP_SRC}/include)
    target_link_libraries(mcnpu PRIVATE llama)
endif()
