#!/bin/sh
set -e
cd "$(dirname "$0")/.."
mkdir -p build
clang++ -std=c++17 -O2 -fobjc-arc -shared -fPIC \
  -framework Metal -framework Foundation \
  src/native/metal_runtime.mm -o build/libmetal_runtime.dylib
echo "built build/libmetal_runtime.dylib"
