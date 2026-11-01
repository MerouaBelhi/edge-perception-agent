cmake_minimum_required(VERSION 3.21)
project(edge_perception_agent_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(MSVC)
  add_compile_options(/O2 /EHsc /utf-8)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

# Path to an unpacked ONNX Runtime release (contains include/ and lib/):
#   https://github.com/microsoft/onnxruntime/releases
#   Linux: onnxruntime-linux-x64-<ver>.tgz   Windows: onnxruntime-win-x64-<ver>.zip
set(ONNXRUNTIME_DIR "$ENV{ONNXRUNTIME_DIR}" CACHE PATH "ONNX Runtime root directory")
if(NOT ONNXRUNTIME_DIR)
  message(FATAL_ERROR "Set ONNXRUNTIME_DIR (env var or -DONNXRUNTIME_DIR=...)")
endif()

# ONNX Runtime as an imported target, so its DLL can be copied next to the exe on Windows
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_DIR}/include")
if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB   "${ONNXRUNTIME_DIR}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/onnxruntime.dll")
elseif(APPLE)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.dylib")
else()
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_DIR}/lib/libonnxruntime.so")
endif()

# Windows prebuilt OpenCV: set OpenCV_DIR to C:/opencv/build
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn)

add_executable(epa_infer main.cpp)
target_include_directories(epa_infer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(epa_infer PRIVATE ${OpenCV_LIBS} onnxruntime)

if(WIN32)
  # Copy onnxruntime.dll and the OpenCV DLLs next to epa_infer.exe. This also stops
  # Windows from loading the older onnxruntime.dll that ships in System32.
  add_custom_command(TARGET epa_infer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:epa_infer> $<TARGET_FILE_DIR:epa_infer>
    COMMAND_EXPAND_LISTS)
  # OpenCV loads its FFmpeg plugin at runtime (needed to read/write .mp4)
  file(GLOB _ffmpeg_dlls "${OpenCV_DIR}/x64/*/bin/opencv_videoio_ffmpeg*.dll"
                         "${OpenCV_DIR}/../bin/opencv_videoio_ffmpeg*.dll"
                         "${OpenCV_DIR}/bin/opencv_videoio_ffmpeg*.dll")
  if(_ffmpeg_dlls)
    add_custom_command(TARGET epa_infer POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_ffmpeg_dlls} $<TARGET_FILE_DIR:epa_infer>)
  else()
    message(WARNING "opencv_videoio_ffmpeg*.dll not found: video files may not open. "
                    "Add the OpenCV bin folder (e.g. C:/opencv/build/x64/vc16/bin) to PATH.")
  endif()
endif()
