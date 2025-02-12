include(FetchContent)

FetchContent_Declare(
  gtest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.13.0
)
FetchContent_MakeAvailable(gtest)

file(GLOB TEST_SOURCES "*.cpp")

foreach(src ${TEST_SOURCES})
    get_filename_component(name ${src} NAME_WE)
    add_executable(${name} ${src})
    target_link_libraries(${name} GTest::gtest GTest::gtest_main)
    target_include_directories(${name} PUBLIC ../include)
endforeach()



