#ifndef MATRIXXXX_H
#define MATRIXXXX_H
#include <fstream>
#include <type_traits>
#include <iomanip>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cooperative_groups.h>
#include <cuda/atomic>
#include <cub/cub.cuh>
#include <cuComplex.h>

#include "cusolver_utils.h"
#include "cuapi.h"


// CUDA API error checking
#define CUDA_CHECK(err)                                                                            \
    do {                                                                                           \
        cudaError_t err_ = (err);                                                                  \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__);                          \
            throw std::runtime_error("CUDA error");                                                \
        }                                                                                          \
    } while (0)

// cusolver API error checking
#define CUSOLVER_CHECK(err)                                                                        \
    do {                                                                                           \
        cusolverStatus_t err_ = (err);                                                             \
        if (err_ != CUSOLVER_STATUS_SUCCESS) {                                                     \
            printf("cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__);                      \
            throw std::runtime_error("cusolver error");                                            \
        }                                                                                          \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                                          \
    do {                                                                                           \
        cublasStatus_t err_ = (err);                                                               \
        if (err_ != CUBLAS_STATUS_SUCCESS) {                                                       \
            printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__);                        \
            throw std::runtime_error("cublas error");                                              \
        }                                                                                          \
    } while (0)

namespace cg = cooperative_groups;


extern const float th13;
constexpr int BLOCK_THREADS = 256;
constexpr int ITEMS_PER_THREAD = 4;
constexpr int TILE_SIZE = BLOCK_THREADS * ITEMS_PER_THREAD;

__constant__ float b[14] = {64764752532480000.,
     32382376266240000.,
     7771770303897600.,
     1187353796428800.,
     129060195264000.,
     10559470521600.,
     670442572800.,
     33522128640.,
     1323241920.,
     40840800.,
     960960.,
     16380.,
     182.,
     1.};


cuComplex cuCexpf(cuComplex z_in);
cuDoubleComplex cuCexp(cuDoubleComplex z_in);



template <typename VT>
__global__ void minus_eye_matrix_trace_kernel(VT *d_A, int32_t n, VT *d_trace) {
    typedef typename cub::BlockReduce<VT, BLOCK_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    VT value = (idx < n) ? d_A[idx * n + idx] : 0.0f;
    

    // Step 2: Block-level reduction
    VT block_sum = BlockReduce(temp_storage).Sum(value);
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        block_sum /= n; // 确保除法使用浮点数
        atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
    }
}

template <typename VT>
struct ComplexSumOp {
    __device__ VT operator()(const VT &a, const VT &b) const {
        if constexpr (std::is_same_v<VT, cuComplex>) {
            return cuCaddf(a, b);
        } else if constexpr (std::is_same_v<VT, cuDoubleComplex>) {
            return cuCadd(a, b);
        } else {
            return a + b;
        }
    }
};


template <>
inline __global__ void minus_eye_matrix_trace_kernel<cuComplex>(cuComplex *d_A, int32_t n, cuComplex *d_trace) {
    typedef typename cub::BlockReduce<cuComplex, BLOCK_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cuComplex value = (idx < n) ? d_A[idx * n + idx] : make_cuComplex(0.0, 0.0);
    

    // Step 2: Block-level reduction
    cuComplex block_sum = BlockReduce(temp_storage).Reduce(value, ComplexSumOp<cuComplex>());
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        // block_sum /= n; // 确保除法使用浮点数
        block_sum.x /= n;
        block_sum.y /= n;
        atomicAdd(&((*d_trace).x), block_sum.x);
        atomicAdd(&((*d_trace).y), block_sum.y);
        // atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        // d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
        d_A[idx * n + idx] = cuCsubf(d_A[idx * n + idx], *d_trace);
    }
}

template <>
inline __global__ void minus_eye_matrix_trace_kernel<cuDoubleComplex>(cuDoubleComplex *d_A, int32_t n, cuDoubleComplex *d_trace) {
    typedef typename cub::BlockReduce<cuDoubleComplex, BLOCK_THREADS> BlockReduce;
    __shared__ BlockReduce::TempStorage temp_storage;
    cg::grid_group grid = cg::this_grid();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    cuDoubleComplex value = (idx < n) ? d_A[idx * n + idx] : make_cuDoubleComplex(0.0, 0.0);
    

    // Step 2: Block-level reduction
    cuDoubleComplex block_sum = BlockReduce(temp_storage).Reduce(value, ComplexSumOp<cuDoubleComplex>());
    __syncthreads();

    // Step 3: First thread in each block writes to global memory
    if (threadIdx.x == 0) {
        // block_sum /= n; // 确保除法使用浮点数
        block_sum.x /= n;
        block_sum.y /= n;
        atomicAdd(&((*d_trace).x), block_sum.x);
        atomicAdd(&((*d_trace).y), block_sum.y);
        // atomicAdd(d_trace, block_sum);
    }

    // Step 4: Grid-level synchronization
    grid.sync();

    // Step 5: Update diagonal
    if (idx < n) {
        // d_A[idx * n + idx] -= *d_trace; // 统一符号，改为减法
        d_A[idx * n + idx] = cuCsub(d_A[idx * n + idx], *d_trace);
    }
}


template <typename VT>
void minus_eye_matrix_trace(VT *d_A, int32_t n, VT *d_trace, cudaStream_t stream) {
    const int BLOCK_THREADS = BLOCK_THREADS;
    dim3 blockSize(BLOCK_THREADS);
    dim3 gridSize((n + BLOCK_THREADS - 1) / BLOCK_THREADS);

    // 检查协作启动支持
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (!deviceProp.cooperativeLaunch) {
        throw std::runtime_error("Device does not support cooperative launch");
    }

    // 初始化 d_trace
    VT trace;
    if constexpr (std::is_same_v<VT, float> || std::is_same_v<VT, double> ) {
        trace = 0.0;
    } else if constexpr (std::is_same_v<VT, cuComplex>) {
        trace = make_cuComplex(0.0, 0.0);
    } else if constexpr (std::is_same_v<VT, cuDoubleComplex>) {
        trace = make_cuDoubleComplex(0.0, 0.0);
    }
    cudaMemcpy(d_trace, &trace, 1 *sizeof(VT), cudaMemcpyHostToDevice);

    // 设置核函数参数
    void *kernelArgs[] = { &d_A, &n, &d_trace };

    // 启动协作核函数
    cudaError_t err = cudaLaunchCooperativeKernel(
        (void*)minus_eye_matrix_trace_kernel<VT>,
        gridSize, blockSize, kernelArgs, 0, stream
    );
    cudaStreamSynchronize(stream);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA Launch Error: " + std::string(cudaGetErrorString(err)));
    }

}


template <typename VT, typename OT> //VT ={float, double, cuComplex, cuDoubleComplex}, OT = {float, double, float, double}
__global__ void RowMaxAbsSumKernel(const VT* __restrict__ A, int n, OT* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<VT, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<VT, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    VT thread_data[ITEMS_PER_THREAD];
    VT thread_sum = 0.0f;

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const VT* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                thread_sum += fabsf(thread_data[i]);
            }
        }
    }

    VT row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<VT, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}

template <>
inline __global__ void RowMaxAbsSumKernel<cuComplex>(const cuComplex* __restrict__ A, int n, float* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<cuComplex, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<float, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    cuComplex thread_data[ITEMS_PER_THREAD];
    float thread_sum = 0.0f; //复数的绝对值都是实数

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const cuComplex* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                // thread_sum += fabsf(thread_data[i]);
                thread_sum += cuCabsf(thread_data[i]);
                // thread_sum.x += thread_data_real;
            }
        }
    }

    float row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<float, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}

template <>
inline __global__ void RowMaxAbsSumKernel<cuDoubleComplex>(const cuDoubleComplex* __restrict__ A, int n, double* max_result) {
    int row = blockIdx.x;
    if (row >= n) return;

    using BlockLoad = cub::BlockLoad<cuDoubleComplex, BLOCK_THREADS, ITEMS_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;
    using BlockReduce = cub::BlockReduce<double, BLOCK_THREADS>;

    __shared__ typename BlockLoad::TempStorage load_temp;
    __shared__ typename BlockReduce::TempStorage reduce_temp;

    cuDoubleComplex thread_data[ITEMS_PER_THREAD];
    double thread_sum = 0.0; //复数的绝对值都是实数

    for (int tile_start = 0; tile_start < n; tile_start += TILE_SIZE) {
        int remaining = n - tile_start;
        int valid_items = min(remaining, TILE_SIZE);
        const cuDoubleComplex* row_ptr = A + row * n + tile_start;

        BlockLoad(load_temp).Load(row_ptr, thread_data, valid_items);
        #pragma unroll
        for (int i = 0; i < ITEMS_PER_THREAD; ++i) {
            int col = tile_start + threadIdx.x * ITEMS_PER_THREAD + i;
            if (col < n) {
                // thread_sum += fabsf(thread_data[i]);
                thread_sum += cuCabs(thread_data[i]);
                // thread_sum.x += thread_data_real;
            }
        }
    }

    double row_sum = BlockReduce(reduce_temp).Sum(thread_sum);

    if (threadIdx.x == 0) {
        cuda::atomic_ref<double, cuda::thread_scope_system> a(*max_result);
        a.fetch_max(row_sum);
    }
}


template <typename VT, typename OT> //VT ={float, double, cuComplex, cuDoubleComplex}, OT = {float, double, float, double}
void RowMaxAbsSum(const VT* d_A, int n, OT* d_max_result, cudaStream_t stream) {
    dim3 blockSize(BLOCK_THREADS);     // 128
    dim3 gridSize(n);                  // 每行一个 block
    RowMaxAbsSumKernel<<<gridSize, blockSize, 0, stream>>>(d_A, n, d_max_result);
}




template <typename VT> //VT = {flaot, double}
__global__ void fuse_kernel( const VT *d_A2, const VT *d_A4, const VT *d_A6,
          VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        VT a6 = d_A6[idx];
        VT a4 = d_A4[idx];
        VT a2 = d_A2[idx];
        VT mask = (idx % (m+1) == 0) ? 1.0f : 0.0f;
        d_u1[idx] = b[13] * a6 + b[11] * a4 + b[9] * a2;
        d_u2[idx] = b[7] * a6 + b[5] * a4 + b[3] * a2 + b[1] * mask;
        d_v1[idx] = b[12] * a6 + b[10] * a4 + b[8] * a2;
        d_v2[idx] = b[6] * a6 + b[4] * a4 + b[2] * a2 + b[0] * mask;
    }
}



template <>
inline __global__ void fuse_kernel<cuComplex>( const cuComplex *d_A2, const cuComplex *d_A4, const cuComplex *d_A6,
          cuComplex *d_u1, cuComplex *d_u2, cuComplex *d_v1, cuComplex *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuComplex a6 = d_A6[idx];
        cuComplex a4 = d_A4[idx];
        cuComplex a2 = d_A2[idx];
        cuComplex mask = (idx % (m+1) == 0) ? make_cuComplex(1.0,0.0) : make_cuComplex(0.0,0.0);
        d_u1[idx].x = b[13] * a6.x + b[11] * a4.x + b[9] * a2.x;
        d_u1[idx].y = b[13] * a6.y + b[11] * a4.y + b[9] * a2.y;

        d_u2[idx].x = b[7] * a6.x + b[5] * a4.x + b[3] * a2.x + b[1] * mask.x;
        d_u2[idx].y = b[7] * a6.y + b[5] * a4.y + b[3] * a2.y + b[1] * mask.y;

        d_v1[idx].x = b[12] * a6.x + b[10] * a4.x + b[8] * a2.x;
        d_v1[idx].y = b[12] * a6.y + b[10] * a4.y + b[8] * a2.y;

        d_v2[idx].x = b[6] * a6.x + b[4] * a4.x + b[2] * a2.x + b[0] * mask.x;
        d_v2[idx].y = b[6] * a6.y + b[4] * a4.y + b[2] * a2.y + b[0] * mask.y;
    }
}

template <>
inline __global__ void fuse_kernel<cuDoubleComplex>( const cuDoubleComplex *d_A2, const cuDoubleComplex *d_A4, const cuDoubleComplex *d_A6,
          cuDoubleComplex *d_u1, cuDoubleComplex *d_u2, cuDoubleComplex *d_v1, cuDoubleComplex *d_v2, int m) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < m * m) {
        cuDoubleComplex a6 = d_A6[idx];
        cuDoubleComplex a4 = d_A4[idx];
        cuDoubleComplex a2 = d_A2[idx];
        cuDoubleComplex mask = (idx % (m+1) == 0) ? make_cuDoubleComplex(1.0,0.0) : make_cuDoubleComplex(0.0,0.0);
        d_u1[idx].x = b[13] * a6.x + b[11] * a4.x + b[9] * a2.x;
        d_u1[idx].y = b[13] * a6.y + b[11] * a4.y + b[9] * a2.y;

        d_u2[idx].x = b[7] * a6.x + b[5] * a4.x + b[3] * a2.x + b[1] * mask.x;
        d_u2[idx].y = b[7] * a6.y + b[5] * a4.y + b[3] * a2.y + b[1] * mask.y;

        d_v1[idx].x = b[12] * a6.x + b[10] * a4.x + b[8] * a2.x;
        d_v1[idx].y = b[12] * a6.y + b[10] * a4.y + b[8] * a2.y;

        d_v2[idx].x = b[6] * a6.x + b[4] * a4.x + b[2] * a2.x + b[0] * mask.x;
        d_v2[idx].y = b[6] * a6.y + b[4] * a4.y + b[2] * a2.y + b[0] * mask.y;
    }
}

template <typename VT>
void fuse( const VT *d_A2, const VT *d_A4, const VT *d_A6, VT *d_u1, VT *d_u2, VT *d_v1, VT *d_v2, int m, cudaStream_t stream) {
    dim3 BlockSize(BLOCK_THREADS);
    dim3 GridSize((m * m+ BlockSize.x -1) / BlockSize.x);
    fuse_kernel<<<GridSize, BlockSize, 0, stream>>>(d_A2, d_A4, d_A6, 
                                  d_u1, d_u2, d_v1, d_v2, m);
}

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, float *d_A,
        float *d_B, float *d_C, float alpha, float beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, double *d_A,
        double *d_B, double *d_C, double alpha, double beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuComplex *d_A,
        cuComplex *d_B, cuComplex *d_C, cuComplex alpha, cuComplex beta);

cublasStatus_t gemm(cublasHandle_t handle, int32_t M, int32_t K, int32_t N, cuDoubleComplex *d_A,
        cuDoubleComplex *d_B, cuDoubleComplex *d_C, cuDoubleComplex alpha, cuDoubleComplex beta);


void solve(cusolverDnHandle_t handle, float *d_A, float *d_B, int m);

void solve(cusolverDnHandle_t handle, double *d_A, double *d_B, int m);

void solve(cusolverDnHandle_t handle, cuComplex *d_A, cuComplex *d_B, int m);

void solve(cusolverDnHandle_t handle, cuDoubleComplex *d_A, cuDoubleComplex *d_B, int m);


template <typename VT>
void readBinaryFloatArray(const std::string& filename, 
                          std::vector<VT>& data, 
                          int numElements) {
    // 打开文件并验证
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件: " + filename);
    }

    // 获取文件大小
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // 验证文件大小是否有效
    size_t fileElements = fileSize / sizeof(VT);
    if (fileSize % sizeof(VT) != 0) {
        throw std::runtime_error("文件大小不匹配: 非浮点数组格式");
    }
    
    // 检查元素数量是否匹配
    if (numElements > 0 && static_cast<size_t>(numElements) != fileElements) {
        throw std::runtime_error("元素数量不匹配: 预期 " + std::to_string(numElements) +
                                " 但实际 " + std::to_string(fileElements));
    }
    
    // 调整向量大小并读取数据
    data.resize(fileElements);
    file.read(reinterpret_cast<char*>(data.data()), fileSize);
    
    // 验证读取完整性
    if (!file || file.gcount() != static_cast<std::streamsize>(fileSize)) {
        throw std::runtime_error("读取不完整: 只读取了 " + 
                                std::to_string(file.gcount()) + "/" + 
                                std::to_string(fileSize) + " 字节");
    }
}

template <typename VT>
void writeBinaryFloatArray(const std::string& filename,
                           const std::vector<VT>& data) {
    // 以二进制写入方式打开文件
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        throw std::runtime_error("无法打开文件进行写入: " + filename);
    }

    // 写入数据
    file.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(VT));

    // 验证写入完整性
    if (!file) {
        throw std::runtime_error("写入失败: " + filename);
    }
}

template <typename VT>
void print_mat(VT *mat, int m, int n, int lda) {
    std::ios_base::fmtflags orig_flags = std::cout.flags();
    int orig_precision = std::cout.precision();
    std::cout << std::fixed << std::setprecision(12);
    for (int i = 0; i < m; ++i) {
        for (int j = 0 ; j < n; ++j) {
            if constexpr(std::is_same_v<VT, float> || std::is_same_v<VT, double>) {
                std::cout << mat[i*lda + j] << ", ";
            } else if constexpr(std::is_same_v<VT, cuComplex>) {
                std::cout << mat[i*lda + j].x << "," << mat[i*lda + j].y << "i" << ",";
            } else if constexpr(std::is_same_v<VT, cuComplex>) {
                std::cout << mat[i*lda + j].x << "," << mat[i*lda + j].y << "i" << ",";
            } 
        }
        std::cout << "\n";
    }
// 恢复 cout 的原始状态
    std::cout.flags(orig_flags);
    std::cout.precision(orig_precision);
}

#endif