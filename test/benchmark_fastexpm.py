import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.linalg import expm
# from fastexpm import expm as cuExpm
import time
import gc
from tqdm import tqdm
import fastexpm


def gpu_time(A, expMat):
    gamma = 1.0
    H = -gamma * A
    psi0 = np.zeros(A.shape[0], dtype=np.complex64)
    psi0[0] = 1.0  # 粒子初始全振幅集中在节点 0

    start = time.perf_counter()
    # print(H.dtype)
    U = expMat.run(-1j * H * 1).reshape(N,N)        # 计算演化算符 U(t)
    psi_t = U.dot(psi0)          # 演化初态
    end = time.perf_counter()
    return end - start


t = 1
N_lt = [100, 1000, 2500, 5000, 8000, 10000]

expMat_Complex64 = fastexpm.MatrixExpCalculator_Complex64(10)

expMat_Complex64.run(np.random.normal(size =(10,10))*1j)
expMat_Complex64.free()
df_gpu_result = pd.DataFrame(columns=["r4", "complete"], index = N_lt, dtype = np.float64)
time.sleep(1)

for N in tqdm(N_lt):
    print(N)
    expMat_Complex64 = fastexpm.MatrixExpCalculator_Complex64(N)
    
    A_r4 = np.fromfile(f"/home/hxy/expm/poster/matrixs/A_r4_{N}_0.bin",dtype = np.complex64).reshape(N,N)
    A_complete = np.fromfile(f"/home/hxy/expm/poster/matrixs/A_complete_{N}_0.bin",dtype = np.complex64).reshape(N,N)


    t = gpu_time(A_r4, expMat = expMat_Complex64)
    df_gpu_result.loc[N,"r4"] = t
    time.sleep(1)


    t = gpu_time(A_complete, expMat = expMat_Complex64)
    df_gpu_result.loc[N,"complete"] = t
    time.sleep(1)
    expMat_Complex64.free()
df_gpu_result.to_csv("df_gpu_result_testv7.csv")

