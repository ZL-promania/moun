cmake_minimum_required(VERSION 3.12)
project(mu_sim_binding)

set(CMAKE_CXX_STANDARD 14)

# 指定 Python 解释器
set(Python_EXECUTABLE "/home/wxy/anaconda3/envs/pyroot-kernel/bin/python")

# 设置 pybind11_DIR
set(pybind11_DIR "/home/wxy/anaconda3/envs/pyroot-kernel/lib/python3.12/site-packages/pybind11/share/cmake/pybind11")

# 通过 pybind11 生成 Python 模块
find_package(pybind11 REQUIRED)

add_library(mu_sim MODULE generate.cpp)



# 去掉库文件的 lib 前缀
set_target_properties(mu_sim PROPERTIES PREFIX "")


target_link_libraries(mu_sim PRIVATE pybind11::module)