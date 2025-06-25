# setup.py (位于 FastCTQW_Project/ 根目录)

import os
import subprocess
import sys
import glob
from pathlib import Path # 导入 Path 模块，用于更友好的路径操作
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext # 导入 setuptools 原始的 build_ext 命令

# --- 定义一些路径和变量 ---
# 获取当前 setup.py 文件所在的目录 (即 FastCTQW_Project/ 根目录)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# CMakeLists.txt 所在的源目录，通常就是项目根目录
CMAKE_SOURCE_DIR = CURRENT_DIR

# Python 主包的名称 (在 FastCTQW_Project/FastCTQW/ 目录)
MAIN_PYTHON_PACKAGE_NAME = 'FastCTQW'
# .so 文件所在的 Python 子模块名 (在 FastCTQW/fastexpm/ 目录)
SO_SUBMODULE_NAME = 'fastexpm'
# .so 文件的通用名称，pybind11 会自动添加类似 .cpython-313-x86_64-linux-gnu.so 的后缀
SO_BASE_FILENAME = '_fastexpm_core'


# --- 定义一个自定义的 Extension 类来承载 CMake 相关的元数据 ---
class CMakeExtension(Extension):
    """
    一个自定义的 Setuptools Extension 类，用于标记一个需要由 CMake 构建的扩展模块。
    Setuptools 不会直接编译这个扩展的源文件，而是会将其传递给我们自定义的 build_ext 命令。
    """
    def __init__(self, name: str, sourcedir: str = CMAKE_SOURCE_DIR) -> None:
        # Extension 的 name 必须是 Python 导入该模块时的完整路径，
        # 例如 'FastCTQW.fastexpm._fastexpm_core'
        # sources=[] 是因为 setuptools 不直接编译这些源文件，而是由 CMake 处理。
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir) # CMakeLists.txt 所在的目录

# --- 定义一个自定义的 build_ext 命令，用于执行 CMake 构建 ---
class CMakeBuild(_build_ext): # 继承自 setuptools 的 build_ext
    """
    自定义的 'build_ext' 命令，用于在 Python 包构建过程中执行 CMake。
    它负责调用 CMake 来配置、编译 C++/CUDA 扩展模块。
    """
    def run(self: _build_ext) -> None:
        # 确保 CMake 已安装并可用
        try:
            subprocess.check_call(['cmake', '--version'])
        except FileNotFoundError:
            raise RuntimeError("CMake must be installed to build the C++/CUDA extension.")

        # 遍历所有被声明的扩展模块
        for ext in self.extensions:
            if isinstance(ext, CMakeExtension):
                # 如果是 CMakeExtension，调用我们自定义的 CMake 构建逻辑
                self.build_cmake_extension(ext)
            else:
                # 对于非 CMakeExtension，调用原始 build_ext 的处理逻辑
                # 这在你的情况下可能不会发生，因为你只有一个 C++ 扩展
                super().build_extension(ext)

    def build_cmake_extension(self, ext: CMakeExtension) -> None:
        """
        执行单个 CMakeExtension 的构建过程 (配置和编译)。
        """
        # extdir 是编译好的 .so 文件应该存放的目录，即 setuptools 的 build/lib.PLATFORM-PYTHONVER 目录。
        # 这是编译后的临时位置，setuptools 会在后续的安装阶段将其复制到 site-packages 中。
        # 例如：/path/to/project/build/lib.linux-x86_64-3.11/FastCTQW/fastexpm/
        extdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()
        extdir.mkdir(parents=True, exist_ok=True) # 确保目标输出目录存在

        # 为 CMake 构建创建一个独立的临时目录
        # 例如：/path/to/project/build/temp.linux-x86_64-3.11/FastCTQW.fastexpm._fastexpm_core/
        build_temp = Path(self.build_temp) / ext.name # 为每个扩展创建独立的构建目录
        build_temp.mkdir(parents=True, exist_ok=True) # 确保 CMake 构建目录存在

        # 根据 Setuptools 的 debug 标志和环境变量设置构建类型 (Release/Debug)
        debug_build = self.debug or int(os.environ.get("DEBUG", 0))
        cfg = 'Debug' if debug_build else 'Release'

        # CMake 配置参数
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}", # CMake 将编译好的库输出到这里
            f"-DPYTHON_EXECUTABLE={sys.executable}",     # 确保 CMake 使用当前的 Python 环境
            f"-DCMAKE_BUILD_TYPE={cfg}",                  # 设置构建类型 (Debug 或 Release)
            # 你可以在这里添加其他特定的 CMake 变量，例如 CUDA 工具包路径等：
            # f"-DCUDA_TOOLKIT_ROOT_DIR={os.environ.get('CUDA_HOME', '')}",
            # '-DPYBIND11_DIR=' + os.path.join(CURRENT_DIR, 'pybind11'), # 如果 pybind11 不是通过 pip 安装的
        ]

        # Windows 操作系统特有的处理：指定 Visual Studio 生成器和平台
        if sys.platform == 'win32':
            # 请根据你的开发环境调整 Visual Studio 的版本，例如 'Visual Studio 17 2022' for VS2022
            cmake_args.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64'])
            cmake_args.append('-DCMAKE_GENERATOR_PLATFORM=x64')

        # 1. 运行 CMake 配置阶段
        print(f"Running CMake config command in {build_temp}: {' '.join(['cmake', ext.sourcedir] + cmake_args)}")
        try:
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        except subprocess.CalledProcessError as e:
            print(f"CMake config failed in {build_temp} with error code {e.returncode}:")
            print(e.output.decode('utf-8') if e.output else "No output.")
            sys.exit(1) # CMake 配置失败，退出构建过程

        # 2. 运行 CMake 构建阶段
        build_command = ['cmake', '--build', '.', '--config', cfg]
        # 添加并行编译选项，使用所有可用 CPU 核心或至少 4 个
        build_command.extend(['-j', str(os.cpu_count() or 4)])

        print(f"Running CMake build command in {build_temp}: {' '.join(build_command)}")
        try:
            subprocess.check_call(build_command, cwd=build_temp)
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed in {build_temp} with error code {e.returncode}:")
            print(e.output.decode('utf-8') if e.output else "No output.")
            sys.exit(1) # CMake 编译失败，退出构建过程

        # 3. 验证 .so 文件是否已生成到预期位置
        # 注意：这里我们检查的是 CMAKE_LIBRARY_OUTPUT_DIRECTORY 指定的路径，
        # 而不是 CMAKE_INSTALL_PREFIX。Setuptools 会自动处理从这里到 site-packages 的复制。
        # pybind11 生成的文件名会包含 ABI 标签和平台信息，所以需要使用 glob 匹配。
        expected_so_path = os.path.join(extdir, f"{SO_BASE_FILENAME}*.so")
        generated_so_files = glob.glob(expected_so_path)

        if not generated_so_files:
            print(f"Error: No .so file matching '{SO_BASE_FILENAME}*.so' found in {extdir} after CMake build.")
            print("Please check your CMakeLists.txt and CMake build output.")
            sys.exit(1)
        else:
            print(f"Successfully built {len(generated_so_files)} .so file(s) to {extdir}")
            for f in generated_so_files:
                print(f"  - {os.path.basename(f)}")

# --- setup() 函数定义你的 Python 包的结构和自定义命令 ---
# 注意：大部分元数据现在都在 pyproject.toml 中定义了，
# setup.py 仅用于 Setuptools 无法直接从 pyproject.toml 读取的复杂逻辑，
# 例如自定义构建 C/C++ 扩展。
setup(
    # 设置 cmdclass 来使用我们自定义的 build_ext 命令
    cmdclass={
        'build_ext': CMakeBuild,
    },
    # 声明一个 Extension，告诉 setuptools 我们有一个 C/C++ 扩展
    # name 必须是 Python 导入该模块时的完整路径，例如 'FastCTQW.fastexpm._fastexpm_core'
    ext_modules=[
        CMakeExtension(f"{MAIN_PYTHON_PACKAGE_NAME}.{SO_SUBMODULE_NAME}.{SO_BASE_FILENAME}")
    ],
    # 启用包发现，通常与 pyproject.toml 中的 [tool.setuptools.packages.find] 结合使用
    # 保留它以确保 setuptools 能够找到所有的 Python 包
    packages=find_packages(where='.'),

    # 重要的：移除所有与 pyproject.toml 重复的元数据定义，以避免警告和配置冲突。
    # 例如，以下这些参数应该从 setup() 调用中删除：
    # name="FastCTQW",
    # version='0.1.0',
    # author='Xiangyang He',
    # author_email='hexiangyangcs@126.com',
    # description='A fast quantum walk package with CUDA acceleration.',
    # long_description=open('README.md', encoding='utf-8').read(),
    # long_description_content_type='text/markdown',
    # url='https://github.com/Crinton/FastCTQW.git',
    # license='BSD 3-Clause',
    # package_data={...}, # 应该由 pyproject.toml 的 [tool.setuptools.package-data] 处理
    # include_package_data=True,
    # classifiers=[...],
    # python_requires='>=3.8',
)
