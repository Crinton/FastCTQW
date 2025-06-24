import subprocess
import sys
import os # 用于检查脚本文件是否存在

def run_benchmark_qwak(script_path, filename, N, graph_type, p , seed):
    """
    运行 script_path 子脚本。

    Args:
        n_value (int): 传递给 --N 参数的值。
        t_value (str): 传递给 --T 参数的值。
    """

    # 检查子脚本文件是否存在
    if not os.path.exists(script_path):
        print(f"错误: 子脚本 '{script_path}' 不存在。请检查路径。")
        return

    # 构建命令行命令列表
    # sys.executable 确保使用当前运行的 Python 解释器来执行子脚本
    command = [
        sys.executable,
        script_path,
        "--f", filename,
        "--N", str(N), # 将整数转换为字符串
        "--T", graph_type,
        "--p", str(p),
        "--seed", str(seed)
    ]

    try:
        # 执行子进程
        # capture_output=False (默认) 表示子进程的 stdout 和 stderr 会直接显示在主进程的控制台上
        # check=True 表示如果子进程返回非零退出码，则抛出 CalledProcessError 异常
        subprocess.run(
            command,
            check=True
        )

    except subprocess.CalledProcessError as e:
        print(f"\n错误: 子脚本 '{script_path}' 执行失败 (返回码: {e.returncode})。")
        print(f"命令: {' '.join(e.cmd)}")
        # 注意: 因为 capture_output=False, 所以这里 e.stdout 和 e.stderr 将是 None
        # 子脚本的输出和错误会直接打印到主进程的控制台
    except FileNotFoundError:
        print(f"错误: Python 解释器或子脚本 '{script_path}' 未找到。")
    except Exception as e:
        print(f"运行子脚本时发生未知错误: {e}")



# filename = "./qwak_cupy_perf.csv"
# N_lt = [100, 1000, 2500, 5000, 8000, 10000]
# # graph_type = "r4" 
# graph_type = "complete"
# p = 0.05
# seed = 0

# for N in N_lt:
#     run_benchmark_qwak("benchmark_qwak_cupy.py", filename, N, graph_type, p, seed)


filename = "./qwak_numpy_perf.csv"
N_lt = [100, 1000, 2500, 5000, 8000, 10000]
# graph_type = "r4" 
graph_type = "complete"
p = 0.05
seed = 0

for N in N_lt:
    run_benchmark_qwak("benchmark_qwak_numpy.py",filename, N, graph_type, p, seed)
