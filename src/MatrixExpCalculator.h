#ifndef MATRIXEXPCALCULATOR_H
#define MATRIXEXPCALCULATOR_H
#include <cmath>       // 包含 expf, cosf, sinf 等数学函数
#include <vector>
#include "matrix.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

template <typename VT>
class MatrixExpCalculator {

// using RT = std::conditional_t<
//     std::is_same_v<VT, cuComplex>, float, // 如果 T 是 cuComplex，结果是 float
//     std::conditional_t<
//         std::is_same_v<VT, cuDoubleComplex>, double, // 否则，如果 T 是 cuDoubleComplex，结果是 double
//         VT // 否则（对于 float, double 或其他），结果是 T 本身
//     >
// >;
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    VT* d_A;
    VT* d_A2;
    VT* d_A4;
    VT* d_A6;
    VT* d_u1;
    VT* d_u2;
    VT* d_v1;
    VT* d_v2;
    VT *d_mu;
    VT *d_nrmA;
    MatrixExpCalculator(size_t n); 

    ~MatrixExpCalculator();

    py::array_t<VT> run(py::array_t<VT>& arr_a);
    
    int32_t getN();

    void free();

};

// MatrixExpCalculator 构造函数的实现
template <typename VT>
MatrixExpCalculator<VT>::MatrixExpCalculator(size_t n) : n(n) {
    /*
    初始化Context资源
    */
    cudaStreamCreate(&stream);
    cublasCreate_v2(&cublasH);
    cusolverDnCreate(&cusolverH);
    cublasSetStream_v2(cublasH, stream);
    cusolverDnSetStream(cusolverH, stream);

    /*
    分配内存
    */

    // 数值算法的必要显存
    cudaMallocAsync(&d_A, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A2, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A4, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_A6, n * n * sizeof(VT), stream);
    cudaMallocAsync(&d_u1, n *n *sizeof(VT), stream);
    cudaMallocAsync(&d_u2, n * n *sizeof(VT), stream);
    cudaMallocAsync(&d_v1, n * n *sizeof(VT), stream);
    cudaMallocAsync(&d_v2, n * n *sizeof(VT), stream);

    // 小显存并初始化
    cudaMallocAsync(&d_mu, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_mu, 0, 1 * sizeof(VT), stream);

    cudaMallocAsync(&d_nrmA, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream);
}

template <typename VT>
MatrixExpCalculator<VT>::~MatrixExpCalculator() {
    free();
}

template <typename VT>
int32_t MatrixExpCalculator<VT>::getN(){
    return this->n;
}

template <typename VT>
void MatrixExpCalculator<VT>::free() {
    /*
    销毁该对象
    */

    if (d_A) {
        CUDA_CHECK(cudaFree(d_A));
        d_A = nullptr;
    };
    if (d_A2) {
        CUDA_CHECK(cudaFree(d_A2));
        d_A2 = nullptr;
    };
    if (d_A4) {
        CUDA_CHECK(cudaFree(d_A4));
        d_A4 = nullptr;
    };
    if (d_A6) {
        CUDA_CHECK(cudaFree(d_A6));
        d_A6 = nullptr;
    };
    if (d_u1) {
        CUDA_CHECK(cudaFree(d_u1));
        d_u1 = nullptr;
    };
    if (d_u2) {
        CUDA_CHECK(cudaFree(d_u2));
        d_u2 = nullptr;
    };
    if (d_v1) {
        CUDA_CHECK(cudaFree(d_v1));
        d_v1 = nullptr;
    };
    if (d_v2) {
        CUDA_CHECK(cudaFree(d_v2));
        d_v2 = nullptr;
    };
    if (d_mu) {
        CUDA_CHECK(cudaFree(d_mu));
        d_mu = nullptr;
    };
    if (d_nrmA) {
        CUDA_CHECK(cudaFree(d_nrmA));
        d_nrmA = nullptr;
    };

    // Destroy cuBLAS handle
    if (cublasH) {
        cublasDestroy_v2(cublasH);
        cublasH = nullptr;
    }
    // Destroy cuSolver handle
    if (cusolverH) {
        cusolverDnDestroy(cusolverH);
        cusolverH = nullptr;
    }
    // Destroy CUDA stream
    if (stream) {
        cudaStreamDestroy(stream);
        stream = nullptr;
    }
}

// MatrixExpCalculator::run 方法的实现
template <typename VT>
py::array_t<VT> MatrixExpCalculator<VT>::run(py::array_t<VT>& arr_a) {
    /*
    每次run从numpy接受一个数组指针
    */
    // cudaEvent_t start, end;
    // cudaEventCreate(&start);
    // cudaEventCreate(&end);
    // cudaEventRecord(start);
    py::buffer_info bufA = arr_a.request();
    auto shape = bufA.shape;
    if (shape[0] != n | shape[1] != n)
    {
        throw std::runtime_error("matrix size is error");
    }
    if (shape[0] != shape[1])
    {
        throw std::runtime_error("Error: The matrix is not square (rows != cols).");
    }
    VT *A = (VT *)bufA.ptr; // 获取 NumPy 数组的原始指针

    VT mu;
    VT nrmA;

    cudaMemcpyAsync(d_A, A, n * n *sizeof(VT), cudaMemcpyHostToDevice,stream);
    
    cudaMemsetAsync(d_mu, 0, 1 * sizeof(VT), stream);
    cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(VT), stream); 

    minus_eye_matrix_trace(d_A, n, d_mu, stream); 
    RowMaxAbsSum(d_A, n, d_nrmA, stream);


    cudaMemcpyAsync(&mu, d_mu, 1 * sizeof(VT), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(VT), cudaMemcpyDeviceToHost);
    int s;
    VT s_pow;
    cudaStreamSynchronize(stream);
    if (nrmA > th13) {
        s = static_cast<int>(std::ceil(std::log2(nrmA / th13))) + 1;
        s_pow = 1 / (static_cast<VT>(std::pow(2,s)));
    } else {
        s = 1;
        s_pow = 0.5f;
    }
    cublasAPI<VT>::Scal(cublasH, n*n, &s_pow, d_A, 1);
    gemm(cublasH, n, n, n, d_A, d_A, d_A2, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, 1.0f, 0.0f);

    gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, 1.0f, 0.0f);

    fuse(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

    gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, 1.0f, 1.0f); //d_u2 = A6 @ u1 + u2

    gemm(cublasH, n, n, n, d_A, d_u2, d_u1, 1.0f, 0.0f); //d_u1 = u = A @ d_u2
    
    gemm(cublasH, n, n, n, d_A6, d_v1, d_v2, 1.0f, 1.0f); //d_v2 = v = A6 @ d_v1 + d_v2

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    VT val_minus_1 = -1.0;
    VT val_plus_1 = 1.0;
    cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
    cublasAPI<VT>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
    cublasAPI<VT>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
    cublasAPI<VT>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2
    solve(cusolverH, d_v1, d_u1, n);
    for (int i = 0; i < s; ++i) {
        gemm(cublasH, n, n, n, d_u1, d_u1, d_v1, 1.0, 0.0);
        cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(VT), cudaMemcpyDeviceToDevice,stream);
    }
    VT emu = std::exp(mu);
    cudaStreamSynchronize(stream);
    cublasAPI<VT>::Scal(cublasH, n*n, &emu, d_u1, 1);
    // compute [13/13] Pade approximant
    VT *eA_ptr = (VT *)malloc(n * n * sizeof(VT));
    
    cudaMemcpy(eA_ptr, d_u1, n * n * sizeof(VT), cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(stream);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error 6: %s, %d\n", cudaGetErrorString(err), __LINE__);
    }
    std::vector<size_t> eA_shape{n*n};
    py::array_t<VT> eA(eA_shape, eA_ptr);

    // float ms;
    // cudaEventRecord(end);
    // cudaEventSynchronize(end);
    // cudaEventElapsedTime(&ms, start, end);
    // printf("GEMM Kernel Time: %.4f ms\n", ms);

    return eA;

}
float get_current_gpu_memory_gb() {
    size_t freeMemBytes, totalMemBytes;
    cudaError_t cudaStatus = cudaMemGetInfo(&freeMemBytes, &totalMemBytes);

    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Error: cudaMemGetInfo failed! " + std::string(cudaGetErrorString(cudaStatus)));
    }

    // 将字节转换为 GB
    const float GB = 1024.0f * 1024.0f * 1024.0f;
    float totalMemGB = static_cast<float>(totalMemBytes) / GB;
    float freeMemGB = static_cast<float>(freeMemBytes) / GB;

    return freeMemGB;
}

template <>
class MatrixExpCalculator<cuComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    cuComplex *d_A;
    cuComplex *d_A2;
    cuComplex *d_A4;
    cuComplex *d_A6;
    cuComplex *d_u1;
    cuComplex *d_u2;
    cuComplex *d_v1;
    cuComplex *d_v2;
    cuComplex *d_mu;
    float *d_nrmA;
    MatrixExpCalculator(size_t n) : n(n) {
        /*
        初始化Context资源
        */
        cudaStreamCreate(&stream);
        cublasCreate_v2(&cublasH);
        cusolverDnCreate(&cusolverH);
        cublasSetStream_v2(cublasH, stream);
        cusolverDnSetStream(cusolverH, stream);
        /*
        分配内存
        */
        // 数值算法的必要显存
        CUDA_CHECK(cudaMallocAsync(&d_A, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A2, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A4, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_A6, n * n * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u1, n *n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_u2, n * n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v1, n * n *sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMallocAsync(&d_v2, n * n *sizeof(cuComplex), stream));

        // 小显存并初始化
        CUDA_CHECK(cudaMallocAsync(&d_mu, 1 * sizeof(cuComplex), stream));
        CUDA_CHECK(cudaMemsetAsync(d_mu, 0, 1 * sizeof(cuComplex), stream));

        CUDA_CHECK(cudaMallocAsync(&d_nrmA, 1 * sizeof(float), stream));
        CUDA_CHECK(cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(float), stream));
    }

    ~MatrixExpCalculator() {
        free();
    }

    py::array_t<std::complex<float>> run(py::array_t<std::complex<float>>& arr_a) {
        py::buffer_info bufA = arr_a.request();
        auto shape = bufA.shape;
        if (shape[0] != n | shape[1] != n)
        {
            throw std::runtime_error("matrix size is error");
        }
        if (shape[0] != shape[1])
        {
            throw std::runtime_error("Error: The matrix is not square (rows != cols).");
        }
        std::complex<float> *A = (std::complex<float> *)bufA.ptr; // 获取 NumPy 数组的原始指针

        cuComplex mu;
        float nrmA;        

        cudaMemcpyAsync(d_A, A, n * n *sizeof(cuComplex), cudaMemcpyHostToDevice,stream);
        

        cudaMemsetAsync(d_mu, 0, 1 * sizeof(cuComplex), stream);
        cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(float), stream); 
        minus_eye_matrix_trace(d_A, n, d_mu, stream); 
        RowMaxAbsSum(d_A, n, d_nrmA, stream);

        cudaMemcpyAsync(&mu, d_mu, 1 * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(float), cudaMemcpyDeviceToHost);

        int s;
        cuComplex s_pow;
        cudaStreamSynchronize(stream);
        if (nrmA > th13) {
            s = static_cast<int>(std::ceil(std::log2(nrmA / th13))) + 1;
            s_pow = make_cuComplex(1/static_cast<float>(std::pow(2,s)),0.0);
        } else {
            s = 1;
            s_pow.x = 0.5;
            s_pow.y = 0.0;
        }
        cublasAPI<cuComplex>::Scal(cublasH, n*n, &s_pow, d_A, 1);
        
        // cudaStreamSynchronize(stream);

        cuComplex alpha_one = make_cuComplex(1.0,0.0);
        cuComplex alpha_zero = make_cuComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one , alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        fuse(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

        gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, alpha_one, alpha_one); //d_u2 = A6 @ u1 + u2

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = u = A @ d_u2

        cuComplex val_minus_1 = make_cuComplex(-1.0, 0.0);
        cuComplex val_plus_1 = make_cuComplex(1.0, 0.0);
        cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
        cublasAPI<cuComplex>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
        cublasAPI<cuComplex>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
        CUBLAS_CHECK(cublasAPI<cuComplex>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1)); // d_u1 = u + v = d_u1 + d_v2

        cudaStreamSynchronize(stream);

        solve(cusolverH, d_v1, d_u1, n);
        for (int i = 0; i < s; ++i) {
            CUBLAS_CHECK(gemm(cublasH, n, n, n, d_u1, d_u1, d_v1,alpha_one, alpha_zero));
            CUDA_CHECK(cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToDevice,stream));
        }
        cuComplex emu = cuCexpf(mu);
        cudaStreamSynchronize(stream);
        CUBLAS_CHECK(cublasAPI<cuComplex>::Scal(cublasH, n*n, &emu, d_u1, 1));
        // compute [13/13] Pade approximant

        std::complex<float> *eA_ptr = (std::complex<float> *)malloc(n * n * sizeof(std::complex<float>));
        
        cudaMemcpyAsync(eA_ptr, d_u1, n * n * sizeof(cuComplex), cudaMemcpyDeviceToHost, stream); // cuComplex -> std::complex<float>
        cudaStreamSynchronize(stream);

        std::vector<size_t> eA_shape{n*n};
        py::array_t<std::complex<float>> eA(eA_shape, eA_ptr);

        // float ms;
        // cudaEventRecord(end);
        // cudaEventSynchronize(end);
        // cudaEventElapsedTime(&ms, start, end);
        // printf("GEMM Kernel Time: %.4f ms\n", ms);

        return eA;
    }
    
    int32_t getN() {
        return this->n;
    }

    void free() {
        /*
        销毁该对象
        */
        if (stream) {
            // 同步 CUDA 流，确保所有待处理的操作（包括之前的异步释放请求）
            // 都已完成。这使得该对象流上的内存释放变为同步操作。
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (d_A) {
            CUDA_CHECK(cudaFree(d_A));
            d_A = nullptr;
        };
        if (d_A2) {
            CUDA_CHECK(cudaFree(d_A2));
            d_A2 = nullptr;
        };
        if (d_A4) {
            CUDA_CHECK(cudaFree(d_A4));
            d_A4 = nullptr;
        };
        if (d_A6) {
            CUDA_CHECK(cudaFree(d_A6));
            d_A6 = nullptr;
        };
        if (d_u1) {
            CUDA_CHECK(cudaFree(d_u1));
            d_u1 = nullptr;
        };
        if (d_u2) {
            CUDA_CHECK(cudaFree(d_u2));
            d_u2 = nullptr;
        };
        if (d_v1) {
            CUDA_CHECK(cudaFree(d_v1));
            d_v1 = nullptr;
        };
        if (d_v2) {
            CUDA_CHECK(cudaFree(d_v2));
            d_v2 = nullptr;
        };
        if (d_mu) {
            CUDA_CHECK(cudaFree(d_mu));
            d_mu = nullptr;
        };
        if (d_nrmA) {
            CUDA_CHECK(cudaFree(d_nrmA));
            d_nrmA = nullptr;
        };

        // Destroy cuBLAS handle
        if (cublasH) {
            cublasDestroy_v2(cublasH);
            cublasH = nullptr;
        }
        // Destroy cuSolver handle
        if (cusolverH) {
            cusolverDnDestroy(cusolverH);
            cusolverH = nullptr;
        }
        // Destroy CUDA stream
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

};

template <>
class MatrixExpCalculator<cuDoubleComplex> {
public:
    size_t n;
    cublasHandle_t cublasH;      // cuBLAS 句柄
    cusolverDnHandle_t cusolverH; // cuSolver 句柄
    cudaStream_t stream;         // CUDA 流
    cuDoubleComplex *d_A;
    cuDoubleComplex *d_A2;
    cuDoubleComplex *d_A4;
    cuDoubleComplex *d_A6;
    cuDoubleComplex *d_u1;
    cuDoubleComplex *d_u2;
    cuDoubleComplex *d_v1;
    cuDoubleComplex *d_v2;
    cuDoubleComplex *d_mu;
    double *d_nrmA;
    MatrixExpCalculator(size_t n) : n(n) {
        /*
        初始化Context资源
        */
        cudaStreamCreate(&stream);
        cublasCreate_v2(&cublasH);
        cusolverDnCreate(&cusolverH);
        cublasSetStream_v2(cublasH, stream);
        cusolverDnSetStream(cusolverH, stream);
        /*
        分配内存
        */

        // 数值算法的必要显存
        cudaMallocAsync(&d_A, n * n * sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_A2, n * n * sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_A4, n * n * sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_A6, n * n * sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_u1, n *n *sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_u2, n * n *sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_v1, n * n *sizeof(cuDoubleComplex), stream);
        cudaMallocAsync(&d_v2, n * n *sizeof(cuDoubleComplex), stream);

        // 小显存并初始化
        cudaMallocAsync(&d_mu, 1 * sizeof(cuDoubleComplex), stream);
        cudaMemsetAsync(d_mu, 0, 1 * sizeof(cuDoubleComplex), stream);

        cudaMallocAsync(&d_nrmA, 1 * sizeof(double), stream);
        cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(double), stream);

    }

    ~MatrixExpCalculator() {
        free();
    }

    py::array_t<std::complex<double>> run(py::array_t<std::complex<double>>& arr_a) {
        py::buffer_info bufA = arr_a.request();
        auto shape = bufA.shape;
        if (shape[0] != n | shape[1] != n)
        {
            throw std::runtime_error("matrix size is error");
        }
        if (shape[0] != shape[1])
        {
            throw std::runtime_error("Error: The matrix is not square (rows != cols).");
        }
        std::complex<double> *A = (std::complex<double> *)bufA.ptr; // 获取 NumPy 数组的原始指针

        cuDoubleComplex mu;
        double nrmA;        
        
        cudaMemcpyAsync(d_A, A, n * n *sizeof(cuDoubleComplex), cudaMemcpyHostToDevice,stream);

        cudaMemsetAsync(d_mu, 0, 1 * sizeof(cuDoubleComplex), stream);
        cudaMemsetAsync(d_nrmA, 0, 1 * sizeof(double), stream); 

        minus_eye_matrix_trace(d_A, n, d_mu, stream); 
        RowMaxAbsSum(d_A, n, d_nrmA, stream);

        cudaMemcpyAsync(&mu, d_mu, 1 * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(&nrmA, d_nrmA, 1 * sizeof(double), cudaMemcpyDeviceToHost);
        int s;
        cuDoubleComplex s_pow;
        cudaStreamSynchronize(stream);
        if (nrmA > th13) {
            s = static_cast<int>(std::ceil(std::log2(nrmA / th13))) + 1;
            s_pow = make_cuDoubleComplex(1/std::pow(2,s),0.0);
        } else {
            s = 1;
            s_pow.x = 0.5;
            s_pow.y = 0.0;
        }
        cublasAPI<cuDoubleComplex>::Scal(cublasH, n*n, &s_pow, d_A, 1);

        cuDoubleComplex alpha_one = make_cuDoubleComplex(1.0,0.0);
        cuDoubleComplex alpha_zero = make_cuDoubleComplex(0.0,0.0);

        gemm(cublasH, n, n, n, d_A, d_A, d_A2, alpha_one , alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A2, d_A4, alpha_one, alpha_zero);

        gemm(cublasH, n, n, n, d_A2, d_A4, d_A6, alpha_one, alpha_zero);

        fuse(d_A2, d_A4, d_A6, d_u1, d_u2, d_v1, d_v2, n, stream);

        gemm(cublasH, n, n, n, d_A6, d_u1, d_u2, alpha_one, alpha_one); //d_u2 = A6 @ u1 + u2

        gemm(cublasH, n, n, n, d_A, d_u2, d_u1, alpha_one, alpha_zero); //d_u1 = u = A @ d_u2

        cuDoubleComplex val_minus_1 = make_cuDoubleComplex(-1.0,0.0);
        cuDoubleComplex val_plus_1 = make_cuDoubleComplex(1.0,0.0);
        cudaMemcpyAsync(d_v1, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice, stream); // d_v1 = d_u1
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n * n, &val_minus_1, d_v2, 1, d_v1, 1); // d_v1 = -u + v = -d_v2 + d_v2
        cublasAPI<cuDoubleComplex>::Scal(cublasH, n * n, &val_minus_1, d_v1, 1);
        cublasAPI<cuDoubleComplex>::Axpy(cublasH, n * n, &val_plus_1, d_v2, 1, d_u1, 1); // d_u1 = u + v = d_u1 + d_v2
        
        cudaStreamSynchronize(stream);

        solve(cusolverH, d_v1, d_u1, n);
        for (int i = 0; i < s; ++i) {
            gemm(cublasH, n, n, n, d_u1, d_u1, d_v1,alpha_one, alpha_zero);
            cudaMemcpyAsync(d_u1, d_v1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToDevice,stream);
        }
        cuDoubleComplex emu = cuCexp(mu);
        cudaStreamSynchronize(stream);
        cublasAPI<cuDoubleComplex>::Scal(cublasH, n*n, &emu, d_u1, 1);
        // compute [13/13] Pade approximant
        std::complex<double> *eA_ptr = (std::complex<double> *)malloc(n * n * sizeof(std::complex<double>));
        
        cudaMemcpy(eA_ptr, d_u1, n * n * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost); // cuComplex -> std::complex<double>
        cudaStreamSynchronize(stream);

        std::vector<size_t> eA_shape{n*n};
        py::array_t<std::complex<double>> eA(eA_shape, eA_ptr);

        // float ms;
        // cudaEventRecord(end);
        // cudaEventSynchronize(end);
        // cudaEventElapsedTime(&ms, start, end);
        // printf("GEMM Kernel Time: %.4f ms\n", ms);

        return eA;
    }
    
    int32_t getN() {
        return this->n;
    }

    void free() {
        /*
        销毁该对象
        */
        if (d_A) {
            CUDA_CHECK(cudaFree(d_A));
            d_A = nullptr;
        };
        if (d_A2) {
            CUDA_CHECK(cudaFree(d_A2));
            d_A2 = nullptr;
        };
        if (d_A4) {
            CUDA_CHECK(cudaFree(d_A4));
            d_A4 = nullptr;
        };
        if (d_A6) {
            CUDA_CHECK(cudaFree(d_A6));
            d_A6 = nullptr;
        };
        if (d_u1) {
            CUDA_CHECK(cudaFree(d_u1));
            d_u1 = nullptr;
        };
        if (d_u2) {
            CUDA_CHECK(cudaFree(d_u2));
            d_u2 = nullptr;
        };
        if (d_v1) {
            CUDA_CHECK(cudaFree(d_v1));
            d_v1 = nullptr;
        };
        if (d_v2) {
            CUDA_CHECK(cudaFree(d_v2));
            d_v2 = nullptr;
        };
        if (d_mu) {
            CUDA_CHECK(cudaFree(d_mu));
            d_mu = nullptr;
        };
        if (d_nrmA) {
            CUDA_CHECK(cudaFree(d_nrmA));
            d_nrmA = nullptr;
        };

        // Destroy cuBLAS handle
        if (cublasH) {
            cublasDestroy_v2(cublasH);
            cublasH = nullptr;
        }
        // Destroy cuSolver handle
        if (cusolverH) {
            cusolverDnDestroy(cusolverH);
            cusolverH = nullptr;
        }
        // Destroy CUDA stream
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

};


#endif