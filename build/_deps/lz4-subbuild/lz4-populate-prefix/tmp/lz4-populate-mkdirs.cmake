# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION ${CMAKE_VERSION}) # this file comes with cmake

# If CMAKE_DISABLE_SOURCE_CHANGES is set to true and the source directory is an
# existing directory in our source tree, calling file(MAKE_DIRECTORY) on it
# would cause a fatal error, even though it would be a no-op.
if(NOT EXISTS "D:/GitHub/CoA_Engine/build/_deps/lz4-src")
  file(MAKE_DIRECTORY "D:/GitHub/CoA_Engine/build/_deps/lz4-src")
endif()
file(MAKE_DIRECTORY
  "D:/GitHub/CoA_Engine/build/_deps/lz4-build"
  "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix"
  "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/tmp"
  "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/src/lz4-populate-stamp"
  "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/src"
  "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/src/lz4-populate-stamp"
)

set(configSubDirs Debug)
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/src/lz4-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "D:/GitHub/CoA_Engine/build/_deps/lz4-subbuild/lz4-populate-prefix/src/lz4-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
