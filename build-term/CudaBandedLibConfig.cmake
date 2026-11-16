
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was CudaBandedLibConfig.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

macro(set_and_check _var _file)
  set(${_var} "${_file}")
  if(NOT EXISTS "${_file}")
    message(FATAL_ERROR "File or directory ${_file} referenced by variable ${_var} does not exist !")
  endif()
endmacro()

macro(check_required_components _NAME)
  foreach(comp ${${_NAME}_FIND_COMPONENTS})
    if(NOT ${_NAME}_${comp}_FOUND)
      if(${_NAME}_FIND_REQUIRED_${comp})
        set(${_NAME}_FOUND FALSE)
      endif()
    endif()
  endforeach()
endmacro()

####################################################################################

# CudaBandedLibConfig.cmake
# Generated config file consumed by downstream projects via:
#   find_package(CudaBandedLib REQUIRED)
#   target_link_libraries(<target> PRIVATE CudaBandedLib::CudaBandedLib)

include(CMakeFindDependencyMacro)

# CudaBandedLib's public link interface pulls in CUDA libraries, so anything
# that links against CudaBandedLib needs CUDAToolkit resolved too.
find_dependency(CUDAToolkit)

# Paths to the installed static library and Fortran .mod files.
set_and_check(CudaBandedLib_LIB_DIR    "${PACKAGE_PREFIX_DIR}/lib")
set_and_check(CudaBandedLib_MODULE_DIR "${PACKAGE_PREFIX_DIR}/include/fortran_modules")

include("${CMAKE_CURRENT_LIST_DIR}/CudaBandedLibTargets.cmake")

check_required_components(CudaBandedLib)
