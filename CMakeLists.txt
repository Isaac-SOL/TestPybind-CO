# CMakeList.txt : projet CMake pour TestPybind, incluez la source et définissez
# la logique spécifique au projet ici.
#
cmake_minimum_required (VERSION 3.8)

project ("TestPybind")

add_subdirectory(extern/pybind11)

include_directories("extern/pybind11/include")

# Ajoutez une source à l'exécutable de ce projet.
add_executable (TestPybind "TestPybind.cpp" "TestPybind.h")
target_link_libraries(TestPybind PRIVATE pybind11::embed)

# TODO: Ajoutez des tests et installez des cibles si nécessaire.
