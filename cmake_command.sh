rm -rf ./build
mkdir ./build
cmake -S . -B build -D CMAKE_INSTALL_PREFIX=./test/fastCTQW/fastexpm
cmake --build build -j 4
cmake --install build