# conftest.py
import pytest
import numpy as np

def pytest_addoption(parser):
    """
    Register custom command-line options.
    注册自定义的命令行选项。
    """
    parser.addoption(
        "--matrix-sizes", # 将参数名改为复数，更清晰
        action="store",
        default="16,32,64,128,256,512,1024", # 默认值，用户可以通过命令行覆盖
        type=str,
        help="逗号分隔的矩阵大小 (例如: --matrix-sizes=500,1000,5000)"
    )
    parser.addoption(
        "--dtype",
        action="store",
        default="complex64", # 默认数据类型
        type=str,          # 明确指定类型为字符串
        help="Data type of random matrices (e.g., --dtype complex64, --dtype complex128)"
    )

def pytest_generate_tests(metafunc):
    """
    根据命令行选项动态参数化测试。
    """
    # 定义字符串数据类型到 NumPy 数据类型的映射
    dtype_map = {
        "complex64": np.complex64,
        "complex128": np.complex128
    }

    # 获取 --matrix-sizes 参数的值
    if "N" in metafunc.fixturenames: # 检查测试函数是否需要 'N' 参数
        matrix_sizes_str = metafunc.config.getoption("--matrix-sizes")
        # 将字符串解析为整数列表
        matrix_sizes = [int(s.strip()) for s in matrix_sizes_str.split(',') if s.strip()]
        
        # 将 matrix_sizes 注入到测试函数的 'N' 参数中
        metafunc.parametrize("N", matrix_sizes)

    # 获取 --dtypes 参数的值
    if "dtype" in metafunc.fixturenames: # 检查测试函数是否需要 'dtype' 参数
        dtypes_str = metafunc.config.getoption("--dtype")
        
        if dtypes_str in dtype_map: # 只有当解析到有效数据类型时才参数化
            metafunc.parametrize("dtype", [dtype_map[dtypes_str]])

    # test-precision 通常不需要用于参数化测试，因为测试函数通常不会根据精度级别重复运行。
    # 它更可能在测试函数内部被访问，以调整断言的容忍度。
    # 如果你确实需要基于 test-precision 参数化，可以按照 N 和 dtype 的方式进行。