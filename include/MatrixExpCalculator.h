#ifndef MATRIXEXPCALCULATOR_H
#define MATRIXEXPCALCULATOR_H

#include <cmath>       // 包含 expf, cosf, sinf 等数学函数
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cuda_runtime.h>  // CUDA 运行时 API
#include <cublas_v2.h>     // cuBLAS API
#include <cusolverDn.h>    // cuSolver API
#include <cuComplex.h>     // cuComplex 类型定义

namespace py = pybind11;



template <typename VT>
class MatrixExpCalculator {
public:
    size_t n;
    cublasHandle_t cublasH;       // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;          // CUDA 流
    VT* d_A;
    VT* d_A2;
    VT* d_A4;
    VT* d_A6;
    VT* d_u1;
    VT* d_u2;
    VT* d_v1;
    VT* d_v2;
    VT *d_mu;
    VT *d_nrmA; // 对于浮点类型，通常是 VT。对于复数类型，是 float/double。

    // 构造函数和析构函数声明
    MatrixExpCalculator(size_t n); 
    ~MatrixExpCalculator();

    // 成员方法声明
    py::array_t<VT> run(py::array_t<VT>& arr_a);
    int32_t getN();
    void free();
};

template <>
class MatrixExpCalculator<cuComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;       // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;          // CUDA 流
    cuComplex* d_A;
    cuComplex* d_A2;
    cuComplex* d_A4;
    cuComplex* d_A6;
    cuComplex* d_u1;
    cuComplex* d_u2;
    cuComplex* d_v1;
    cuComplex* d_v2;
    cuComplex *d_mu;
    float *d_nrmA; // 对于浮点类型，通常是 VT。对于复数类型，是 float/double。

    // 构造函数和析构函数声明
    MatrixExpCalculator(size_t n); 
    ~MatrixExpCalculator();

    // 成员方法声明
    py::array_t<std::complex<float>> run(py::array_t<std::complex<float>>& arr_a);
    int32_t getN();
    void free();
};

template <>
class MatrixExpCalculator<cuDoubleComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;       // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;          // CUDA 流
    cuDoubleComplex* d_A;
    cuDoubleComplex* d_A2;
    cuDoubleComplex* d_A4;
    cuDoubleComplex* d_A6;
    cuDoubleComplex* d_u1;
    cuDoubleComplex* d_u2;
    cuDoubleComplex* d_v1;
    cuDoubleComplex* d_v2;
    cuDoubleComplex *d_mu;
    double *d_nrmA; // 对于浮点类型，通常是 VT。对于复数类型，是 float/double。

    // 构造函数和析构函数声明
    MatrixExpCalculator(size_t n); 
    ~MatrixExpCalculator();

    // 成员方法声明
    py::array_t<std::complex<double>> run(py::array_t<std::complex<double>>& arr_a);
    int32_t getN();
    void free();
};

float get_current_gpu_memory_gb();

#endif // MATRIXEXPCALCULATOR_H