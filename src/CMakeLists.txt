add_executable(clifford main.cpp)

#set_property(TARGET clifford PROPERTY INTERPROCEDURAL_OPTIMIZATION True)
set_property(TARGET clifford PROPERTY CXX_STANDARD 20)

find_package (Python COMPONENTS Interpreter Development NumPy)

if (${Python_FOUND})
    message(STATUS "Python found")
    message(STATUS ${Python_VERSION})
else()
    message(SEND_ERROR "Python not found")
endif()

target_include_directories(clifford PRIVATE ${Python_INCLUDE_DIRS})
