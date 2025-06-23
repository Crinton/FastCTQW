#include "matrix.h"
const float th13 = 5.37;
cuComplex cuCexpf(cuComplex z) {

    float x = z.x;
    float y = z.y;
    float exp_x = expf(x);
    float cos_y = cosf(y);
    float sin_y = sinf(y);
    return make_cuComplex(exp_x * cos_y, exp_x * sin_y);
}

cuDoubleComplex cuCexp(cuDoubleComplex z) {

    double x = z.x;
    double y = z.y;
    double exp_x = expf(x);
    double cos_y = cosf(y);
    double sin_y = sinf(y);
    return make_cuDoubleComplex(exp_x * cos_y, exp_x * sin_y);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, float *d_A, float *d_B, float *d_C, float alpha, float beta) {
    /*
    d_A (MxK), d_B(KxN), d_C(MxN) row-major
    */ 
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    return cublasSgemm_v2(handle,
                  transa, transb, 
                  N, M, K, 
                  &alpha, 
                  d_B, N,
                  d_A, K,
                  &beta,
                  d_C, N);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, double *d_A,
        double *d_B, double *d_C, double alpha, double beta) {
    /*
    d_A (MxK), d_B(KxN), d_C(MxN) row-major
    */ 
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    return cublasDgemm_v2(handle,
                  transa, transb, 
                  N, M, K, 
                  &alpha, 
                  d_B, N,
                  d_A, K,
                  &beta,
                  d_C, N);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuComplex *d_A,
        cuComplex *d_B, cuComplex *d_C, cuComplex alpha, cuComplex beta) {
    /*
    d_A (MxK), d_B(KxN), d_C(MxN) row-major
    */ 
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    return cublasCgemm_v2(handle,
                  transa, transb, 
                  N, M, K, 
                  &alpha, 
                  d_B, N,
                  d_A, K,
                  &beta,
                  d_C, N);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuDoubleComplex *d_A, cuDoubleComplex *d_B, cuDoubleComplex *d_C, cuDoubleComplex alpha, cuDoubleComplex beta) {
    /*
    d_A (MxK), d_B(KxN), d_C(MxN) row-major
    */ 
    cublasOperation_t transa = CUBLAS_OP_N;
    cublasOperation_t transb = CUBLAS_OP_N;
    // float alpha = 1.0f;
    // float beta = 0.0f;
    return cublasZgemm_v2(handle,
                  transa, transb, 
                  N, M, K, 
                  &alpha, 
                  d_B, N,
                  d_A, K,
                  &beta,
                  d_C, N);
}

void solve(cusolverDnHandle_t handle,float *d_A, float *d_B, int m) {
    using data_type = float;
    const int64_t lda = m;
    const int64_t ldb = m;
    
    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */

    const int algo = 0;
    /* Create advanced params */
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    if (algo == 0) {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    } else {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1);
    }
    cusolverDnXgetrf_bufferSize(handle, params, m, m, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice,
                                    &workspaceInBytesOnHost);
    cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
    cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * m);
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    /* step 4: LU factorization */

    cusolverDnXgetrf(handle, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info);
    cusolverDnXgetrs(handle, params, CUBLAS_OP_N, m, m, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info);
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);
}


void solve(cusolverDnHandle_t handle, double *d_A, double *d_B, int m) {
    using data_type = double;
    const int64_t lda = m;
    const int64_t ldb = m;

    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */

    const int algo = 0;
    /* Create advanced params */
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    if (algo == 0) {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    } else {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1);
    }
    cusolverDnXgetrf_bufferSize(handle, params, m, m, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice,
                                    &workspaceInBytesOnHost);
    cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
    cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * m);
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    /* step 4: LU factorization */

    cusolverDnXgetrf(handle, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info);
    cusolverDnXgetrs(handle, params, CUBLAS_OP_N, m, m, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info);
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);
}

void solve(cusolverDnHandle_t handle, cuComplex *d_A, cuComplex *d_B, int m) {
    using data_type = cuComplex;
    const int64_t lda = m;
    const int64_t ldb = m;

    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */

    const int algo = 0;
    /* Create advanced params */
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    if (algo == 0) {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    } else {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1);
    }
    cusolverDnXgetrf_bufferSize(handle, params, m, m, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice,
                                    &workspaceInBytesOnHost);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * m));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    /* step 4: LU factorization */
    CUSOLVER_CHECK(cusolverDnXgetrf(handle, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info));
    CUSOLVER_CHECK(cusolverDnXgetrs(handle, params, CUBLAS_OP_N, m, m, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info));
    cudaDeviceSynchronize();

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);
}


void solve(cusolverDnHandle_t handle, cuDoubleComplex *d_A, cuDoubleComplex *d_B, int m) {
    using data_type = cuDoubleComplex;
    const int64_t lda = m;
    const int64_t ldb = m;

    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t workspaceInBytesOnDevice = 0; /* size of workspace */
    void *d_work = nullptr;              /* device workspace for getrf */
    size_t workspaceInBytesOnHost = 0;   /* size of workspace */
    void *h_work = nullptr;              /* host workspace for getrf */

    const int algo = 0;
    /* Create advanced params */
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    if (algo == 0) {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
    } else {
        cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1);
    }
    cusolverDnXgetrf_bufferSize(handle, params, m, m, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice,
                                    &workspaceInBytesOnHost);
    cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
    cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * m);
    cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
    if (0 < workspaceInBytesOnHost) {
        h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
        if (h_work == nullptr) {
            throw std::runtime_error("Error: h_work not allocated.");
        }
    }
    /* step 4: LU factorization */

    cusolverDnXgetrf(handle, params, m, m, traits<data_type>::cuda_data_type,
                                        d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                        workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info);
    cusolverDnXgetrs(handle, params, CUBLAS_OP_N, m, m, /* nrhs */
                                        traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                        traits<data_type>::cuda_data_type, d_B, ldb, d_info);
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_work));
    free(h_work);
}


// void solve(cusolverDnHandle_t handle,float *d_A, float *d_B, int m, void *d_work, void *h_work, int64_t *d_Ipiv) {
//     using data_type = float;
//     const int64_t lda = m;
//     const int64_t ldb = m;

//     int64_t *d_Ipiv = nullptr; /* pivoting sequence */
//     int *d_info = nullptr;     /* error info */

//     size_t workspaceInBytesOnDevice = 0; /* size of workspace */
//     void *d_work = nullptr;              /* device workspace for getrf */
//     size_t workspaceInBytesOnHost = 0;   /* size of workspace */
//     void *h_work = nullptr;              /* host workspace for getrf */

//     const int algo = 0;
//     /* Create advanced params */
//     cusolverDnParams_t params;
//     cusolverDnCreateParams(&params);
//     if (algo == 0) {
//         cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0);
//     } else {
//         cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_1);
//     }
//     cusolverDnXgetrf_bufferSize(handle, params, m, m, traits<data_type>::cuda_data_type, d_A,
//                                     lda, traits<data_type>::cuda_data_type, &workspaceInBytesOnDevice,
//                                     &workspaceInBytesOnHost);
//     cudaMalloc(reinterpret_cast<void **>(&d_work), workspaceInBytesOnDevice);
//     cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * m);
//     cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
//     if (0 < workspaceInBytesOnHost) {
//         h_work = reinterpret_cast<void *>(malloc(workspaceInBytesOnHost));
//         if (h_work == nullptr) {
//             throw std::runtime_error("Error: h_work not allocated.");
//         }
//     }
//     /* step 4: LU factorization */

//     cusolverDnXgetrf(handle, params, m, m, traits<data_type>::cuda_data_type,
//                                         d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
//                                         workspaceInBytesOnDevice, h_work, workspaceInBytesOnHost, d_info);
//     cusolverDnXgetrs(handle, params, CUBLAS_OP_N, m, m, /* nrhs */
//                                         traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
//                                         traits<data_type>::cuda_data_type, d_B, ldb, d_info);
//     CUDA_CHECK(cudaFree(d_info));
//     // CUDA_CHECK(cudaFree(d_Ipiv));
//     // CUDA_CHECK(cudaFree(d_work));
//     // free(h_work);
//     cusolverDnDestroyParams(params);

// }
