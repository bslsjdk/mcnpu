cmake_minimum_required(VERSION 3.22.1)
project(mcnpu)
add_library(mcnpu SHARED mcnpu.cpp)
set(QNN_INC "" CACHE PATH "QNN headers")
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
target_compile_definitions(mcnpu PRIVATE MCNPU_BUILD_ID="${MCNPU_BUILD_ID}")
