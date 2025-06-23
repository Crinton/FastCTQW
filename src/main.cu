#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "MatrixExpCalculator.h"
namespace py = pybind11;



PYBIND11_MODULE(fastexpm, m) {
    m.doc() = "pybind11 example plugin for matrix exponential"; // 模块的文档字符串

    // 绑定expm_float函数到Python模块
    py::class_<MatrixExpCalculator<float>>(m, "ExpMatFloat32")
        // 绑定构造函数，接受一个 int 参数 n
        .def(py::init<size_t>(), py::arg("n"),
             "Initializes the MatrixExpCalculator for n x n matrices, allocating GPU memory.")
        // 绑定 run 方法
        .def("run", &MatrixExpCalculator<float>::run,
             py::arg("arr_a"), // run 现在也接受 arr_a 作为输入
             "Performs matrix exponential calculation on the given NumPy array.")
        // 主动释放的内存
        .def("free", &MatrixExpCalculator<float>::free,
             "Free memory")
        ;

     py::class_<MatrixExpCalculator<double>>(m, "ExpMatFloat64")
        // 绑定构造函数，接受一个 int 参数 n
        .def(py::init<size_t>(), py::arg("n"),
             "Initializes the MatrixExpCalculator for n x n matrices, allocating GPU memory.")
        // 绑定 run 方法
        .def("run", &MatrixExpCalculator<double>::run,
             py::arg("arr_a"), // run 现在也接受 arr_a 作为输入
             "Performs matrix exponential calculation on the given NumPy array.")
        // 主动释放的内存
        .def("free", &MatrixExpCalculator<double>::free,
             "Free memory")
        ;

     py::class_<MatrixExpCalculator<cuComplex>>(m, "ExpMatComplex64")
        // 绑定构造函数，接受一个 int 参数 n
        .def(py::init<size_t>(), py::arg("n"),
             "Initializes the MatrixExpCalculator for n x n matrices, allocating GPU memory.")
        // 绑定 run 方法
        .def("run", &MatrixExpCalculator<cuComplex>::run,
             py::arg("arr_a"), // run 现在也接受 arr_a 作为输入
             "Performs matrix exponential calculation on the given NumPy array.")
        // 主动释放的内存
        .def("free", &MatrixExpCalculator<cuComplex>::free,
             "Free memory")
        ;

     py::class_<MatrixExpCalculator<cuDoubleComplex>>(m, "ExpMatComplex128")
        // 绑定构造函数，接受一个 int 参数 n
        .def(py::init<size_t>(), py::arg("n"),
             "Initializes the MatrixExpCalculator for n x n matrices, allocating GPU memory.")
        // 绑定 run 方法
        .def("run", &MatrixExpCalculator<cuDoubleComplex>::run,
             py::arg("arr_a"), // run 现在也接受 arr_a 作为输入
             "Performs matrix exponential calculation on the given NumPy array.")
        // 主动释放的内存
        .def("free", &MatrixExpCalculator<cuDoubleComplex>::free,
             "Free memory")
        ;

}

