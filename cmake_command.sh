rm -rf ./build
mkdir ./build
cmake -S . -B build -D CMAKE_INSTALL_PREFIX=./test/fastCTQW/fastexpm
cmake --build build -j 4
cmake --install build

nvcc -c cuapi.cu -o cuapi.o -I./ -std=c++17  -Xcompiler -fPIC -gencode arch=compute_89,code=sm_89
nvcc -c matrix.cu -o matrix.o -I./ -std=c++17  -Xcompiler -fPIC -gencode arch=compute_89,code=sm_89
nvcc -c MatrixExpCalculator.cu -o MatrixExpCalculator.o -I ../pybind11/include -I./  -I /home/hxy/anaconda3/envs/QWAK_env/include/python3.13/ -Xcompiler -fPIC -std=c++17 -gencode arch=compute_89,code=sm_89
ar rcs libmatrixexp.a matrix.o MatrixExpCalculator.o