#!/bin/bash
CC=~/Projects/clang+llvm-14.0.0-x86_64-apple-darwin/bin/clang++
"$CC" exports.cpp --target=wasm32 -o exports.wasm -Wall -Wl,--no-entry -Wl,--export-all -std=c++20 -nostdlib -O3 -flto -Wl,--lto-O3 -ffreestanding
wasm-opt -O4 exports.wasm -o exports.wasm.opt
