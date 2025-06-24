# setup.py (位于 FastCTQW_Project/ 根目录)

import os
import subprocess
import sys
import glob # 导入 glob 模块用于文件查找
from setuptools import setup, find_packages
from setuptools.command.install import install as _install

# --- 定义一些路径和变量 ---
# 获取当前 setup.py 文件所在的目录 (即 FastCTQW_Project/ 根目录)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))

# CMake 构建的临时目录，通常是项目根目录下的 'build/'
CMAKE_BUILD_DIR = os.path.join(CURRENT_DIR, 'build')

# Python 主包的名称 (在 FastCTQW_Project/fastctqw/ 目录)
MAIN_PYTHON_PACKAGE_NAME = 'FastCTQW' # 注意：这里是大写，与你的文件夹名一致
# .so 文件所在的 Python 子模块名 (在 fastctqw/fastexpm/ 目录)
SO_SUBMODULE_NAME = 'fastexpm' # 注意：这里是小写，与你的文件夹名一致
# .so 文件的通用名称，pybind11 会自动添加类似 .cpython-313-x86_64-linux-gnu.so 的后缀
SO_BASE_FILENAME = '_fastexpm_core'

# --- 定义一个自定义的安装命令，用于在 pip 安装前执行 CMake 构建 ---
class CMakeBuildInstall(_install):
    """
    自定义的 'install' 命令，用于在 Python 包安装前执行 CMake 构建。
    这确保了 .so 文件在安装到 site-packages 之前被正确编译和放置。
    """
    def run(self: _install): # 显式类型提示 _install
        # 1. 确保 CMake 构建目录存在
        if not os.path.exists(CMAKE_BUILD_DIR):
            os.makedirs(CMAKE_BUILD_DIR)

        # 2. 确定 CMake 的安装前缀 (CMAKE_INSTALL_PREFIX)
        # 这个前缀应该指向 Python 包在目标系统上的最终安装目录，
        # 例如: /path/to/venv/lib/pythonX.Y/site-packages/FastCTQW/fastexpm/
        # self.install_lib 提供了 site-packages 的路径。
        so_install_target_path  = os.path.join(self.install_lib, MAIN_PYTHON_PACKAGE_NAME, SO_SUBMODULE_NAME)
        print(f"CMake will install .so to: {so_install_target_path }")

        # 确保这个路径是绝对路径，且存在
        os.makedirs(so_install_target_path, exist_ok=True)
        so_install_target_path = os.path.abspath(so_install_target_path)
        # 3. 构建 CMake 配置命令
        cmake_config_command = [
            'cmake',
            CURRENT_DIR,  # CMakeLists.txt 所在的源目录
            f'-DPYTHON_EXECUTABLE={sys.executable}', # 确保 CMake 使用当前的 Python 环境
            f'-DCMAKE_INSTALL_PREFIX={so_install_target_path }', # 设置 CMake 的安装前缀
            # 如果需要，可以在这里添加其他 CMake 变量，例如自定义 Pybind11 路径
            # '-DPYBIND11_DIR=' + os.path.join(CURRENT_DIR, 'pybind11'),
        ]

        # 根据操作系统设置构建类型，通常打包发布用 Release
        if sys.platform.startswith('linux') or sys.platform == 'darwin': # Linux / macOS
            cmake_config_command.append('-DCMAKE_BUILD_TYPE=Release')
        elif sys.platform == 'win32': # Windows
            # 对于 Windows，需要指定生成器和平台
            # 示例：'Visual Studio 17 2022'，根据用户环境的 Visual Studio 版本调整
            cmake_config_command.extend(['-G', 'Visual Studio 17 2022', '-A', 'x64'])
            # 显式指定平台
            cmake_config_command.append('-DCMAKE_GENERATOR_PLATFORM=x64')

        print(f"Running CMake config command: {' '.join(cmake_config_command)}")
        try:
            # 在 CMake 构建目录中执行配置命令
            subprocess.check_call(cmake_config_command, cwd=CMAKE_BUILD_DIR)
        except subprocess.CalledProcessError as e:
            print(f"CMake config failed with error code {e.returncode}: {e.output.decode('utf-8') if e.output else ''}")
            sys.exit(1) # CMake 配置失败，退出安装

        # 4. 构建 CMake 编译命令
        cmake_build_command = [
            'cmake',
            '--build', CMAKE_BUILD_DIR, # 指定构建目录
            '--config', 'Release', # 对于多配置生成器 (如 Visual Studio)，需要指定 --config
            '-j', str(os.cpu_count() or 4) # 使用 CPU 核心数进行并行编译，至少 4
        ]
        print(f"Running CMake build command: {' '.join(cmake_build_command)}")
        try:
            subprocess.check_call(cmake_build_command)
        except subprocess.CalledProcessError as e:
            print(f"CMake build failed with error code {e.returncode}: {e.output.decode('utf-8') if e.output else ''}")
            sys.exit(1) # CMake 编译失败，退出安装

        # 5. 运行 CMake install 命令
        # 这会将编译好的 .so 文件复制到 CMAKE_INSTALL_PREFIX 指定的路径
        cmake_install_command = [
            'cmake',
            '--install', CMAKE_BUILD_DIR # 指定从哪个构建目录进行安装
        ]
        print(f"Running CMake install command: {' '.join(cmake_install_command)}")
        try:
            subprocess.check_call(cmake_install_command)
        except subprocess.CalledProcessError as e:
            print(f"CMake install failed with error code {e.returncode}: {e.output.decode('utf-8') if e.output else ''}")
            sys.exit(1) # CMake 安装失败，退出安装

        # 6. 验证 .so 文件是否已成功安装到目标 Python 包目录
        # pybind11 生成的文件名会包含 ABI 标签和平台信息，所以需要使用 glob 匹配
        # 例如 _fastexpm_core.cpython-313-x86_64-linux-gnu.so
        installed_so_files = glob.glob(os.path.join(so_install_target_path, f"{SO_BASE_FILENAME}*.so"))
        if not installed_so_files:
            print(f"Error: {SO_BASE_FILENAME}*.so not found at {so_install_target_path} after CMake install.")
            print("Please check your CMakeLists.txt 'install' rule and setup.py paths.")
            sys.exit(1)
        else:
            print(f"Successfully installed {len(installed_so_files)} .so file(s) to {so_install_target_path}")
            for f in installed_so_files:
                print(f"  - {os.path.basename(f)}")

        # 7. 调用原始的 install 命令来完成 Python 包的其余安装工作
        # 此时，.so 文件已经被 CMake 放置到正确的安装位置，setuptools 会发现它
        super().run()

# --- setup() 函数定义你的 Python 包的元数据和内容 ---
setup(
    name= "FastCTQW", # 你的 Python 包名，用户通过 `pip install FastCTQW` 安装
    version='0.1.0', # 你的包版本，每次发布新版本时记得更新
    author='Xiangyang He', # 作者名
    author_email='hexiangyangcs@126.com', 
    description='A fast quantum walk package with CUDA acceleration.', # 包的简短描述
    long_description=open('README.md', encoding='utf-8').read(), # 包的详细描述，通常从 README.md 读取
    long_description_content_type='text/markdown', # 详细描述的内容类型
    url='https://github.com/Crinton/FastCTQW.git', # 你的 GitHub 仓库 URL
    license='BSD 3-Clause', # 你的许可证类型，例如 'MIT', 'Apache-2.0' 等

    # find_packages() 会在当前目录 (.) 下查找所有包含 __init__.py 的目录
    # 这将找到 'FastCTQW' 和 'FastCTQW.fastexpm'
    packages=find_packages(where='.'),

    # 明确告诉 setuptools 包含 .so 文件
    # 注意这里，package_data 的键是完整的 Python 包导入路径，值是文件名列表。
    # 我们使用通配符来匹配 Pybind11 生成的带有 ABI 标签的文件名。
    package_data={
        f'{MAIN_PYTHON_PACKAGE_NAME}.{SO_SUBMODULE_NAME}': [f'{SO_BASE_FILENAME}*.so'],
        # 如果你有其他非 Python 文件需要包含在特定子包中，也在这里列出
        # 例如：'FastCTQW': ['*.txt', 'some_data_file.json'],
    },
    include_package_data=True, # 确保 package_data 中指定的文件被包含进来

    # 包的分类器，帮助用户在 PyPI 上发现你的包
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13', # 根据你的支持版本调整
        'Programming Language :: Python :: 3.14', # 根据你的支持版本调整
        'License :: OSI Approved :: BSD 3-Clause License',
        'Operating System :: POSIX :: Linux',
        'Development Status :: 4 - Beta', # 3 - Alpha (开发中), 4 - Beta (测试版), 5 - Production/Stable (稳定版)
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        "Topic :: Scientific/Engineering :: Quantum",
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8', 
    install_requires=[
        'numpy>=1.20', 
        'networkx>=3.0',
        'scipy>=1.14',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0'
    ],
    
    cmdclass={
        'install': CMakeBuildInstall,
    },
)