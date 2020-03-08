# Copyright 2020 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(mnist)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS})

add_executable(train
        src/train.cpp
        src/utils.cpp
        )
target_link_libraries(train ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET train PROPERTY CXX_STANDARD 20)

add_executable(test
        src/test.cpp
        src/utils.cpp
        )
target_link_libraries(test ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET test PROPERTY CXX_STANDARD 20)

add_executable(detect
        src/detect.cpp
        src/classifier.cpp
        src/utils.cpp
        )
target_link_libraries(detect ${TORCH_LIBRARIES} ${OpenCV_LIBS})
set_property(TARGET detect PROPERTY CXX_STANDARD 20)