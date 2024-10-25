# A tree-based model compiler based on MLIR
## How to build
```bash
git clone https://github.com/njru8cjo/tree-house
mkdir build && cd build
cmake ..
cmake --build .
```

## Dump LLVM IR code
```bash
./build/bin/frontend --dump
```

## Pre-request
MLIR build with llvm-project 18.0.1
