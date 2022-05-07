cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(symcuda LANGUAGES CUDA CXX)

file(GLOB SYMCUDA_SRC
	"*.cu"
	"*.cuh"
)

# add_library(symcuda SHARED ${SYMCUDA_SRC})
add_executable(symcuda ${SYMCUDA_SRC})
# add_link_options(-lm -g -G -arch=sm_37 -Wno-deprecated-gpu-targets)
# add_compile_options(-MD -arch=sm_37 -Wno-deprecated-gpu-targets -I. -rdc=true)

set_target_properties(symcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_compile_options(symcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                      	--generate-line-info
                      	--relocatable-device-code=true
												-Wno-deprecated-gpu-targets
												-arch=sm_37
												-MD
												-I.
                       >)

target_link_options(symcuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
												-lm
												-lcudart
												-lcudadevrt
                       >)


set_target_properties(symcuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")

# file(GLOB SYMCUDA_SRC
# 	"*.h"
# 	"*.cpp"
# 	"*.cu"
# 	"*.cuh"
# )
# 
# set_target_properties(symcuda PROPERTIES CUDA_ARCHITECTURES "35;50;72")
# 
# # Request that particles be built with -std=c++11
# # As this is a public compile feature anything that links to 
# # particles will also build with -std=c++11
# target_compile_features(symcuda PUBLIC cxx_std_11)
#  
# # We need to explicitly state that we need all CUDA files in the 
# # particle library to be built with -dc as the member functions 
# # could be called by other libraries and executables
# set_target_properties( symcuda
#                        PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
#  
#  
# set_property(TARGET symcuda 
#              PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# add_executable(main main.cpp)
# target_link_libraries(main PRIVATE symcuda)
# target_link_libraries(main ${CUDA_LIBRARIES})
# set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "35;50;72")

# 
# add_executable(main ${SYMCUDA_SRC})
