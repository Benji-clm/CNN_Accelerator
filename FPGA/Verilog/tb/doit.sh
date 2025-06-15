#!/bin/bash

# This script runs the testbench
# Usage: ./doit.sh <file1.cpp> <file2.cpp>

# Constants
SCRIPT_DIR=$(dirname "$(realpath "$0")")
TEST_FOLDER=$(realpath "$SCRIPT_DIR/tests")
ROOT_RTL_FOLDER=$(realpath "$SCRIPT_DIR/../rtl")
GREEN=$(tput setaf 2)
RED=$(tput setaf 1)
RESET=$(tput sgr0)

# Variables
passes=0
fails=0

# Handle terminal arguments
if [[ $# -eq 0 ]]; then
    # If no arguments provided, run all tests
    files=(${TEST_FOLDER}/*.cpp)
else
    # If arguments provided, use them as input files
    files=("$@")
fi

# Cleanup
rm -rf obj_dir

cd $SCRIPT_DIR

# Iterate through files
for file in "${files[@]}"; do
   # Extract the directory name of the test file
    TEST_FILE_DIR=$(dirname "$file")
    TEST_SUBDIR=$(basename "$TEST_FILE_DIR")
    echo "TEST_SUBDIR: $TEST_SUBDIR"
    # Set RTL_FOLDER based on TEST_SUBDIR
    if [ "$TEST_SUBDIR" == "tests" ]; then
        RTL_FOLDER=$(realpath "$SCRIPT_DIR/../rtl")
    else
        RTL_FOLDER="${SCRIPT_DIR}/../rtl/${TEST_SUBDIR}"
    fi
    echo "RTL_FOLDER: $RTL_FOLDER"

    name=$(basename "$file" _tb.cpp | cut -f1 -d\-)
    echo "Name: $name"
    # If verify.cpp -> we are testing the top module
    if [ $name == "verify.cpp" ]; then
        name="top"
    fi

    # Translate Verilog -> C++ including testbench
    verilog_file=$(find "${RTL_FOLDER}" -type f -name "${name}.sv" | head -n 1)
    
    # Build include paths for all RTL subdirectories
    INCLUDE_PATHS="-y ${ROOT_RTL_FOLDER}"
    for dir in $(find "${ROOT_RTL_FOLDER}" -type d); do
        if [ "$dir" != "${ROOT_RTL_FOLDER}" ]; then
            INCLUDE_PATHS="${INCLUDE_PATHS} -y $dir"
        fi
    done
    
    verilator   -Wall --trace \
                -cc "${verilog_file}" \
                --exe "${file}" \
                ${INCLUDE_PATHS} \
                --prefix "Vdut" \
                -o Vdut \
                -LDFLAGS "-lgtest -lgtest_main -lpthread -lImath" \

    # Build C++ project with automatically generated Makefile
    make -j -C obj_dir/ -f Vdut.mk
    
    # Run executable simulation file
    ./obj_dir/Vdut
    
    # Check if the test succeeded or not
    if [ $? -eq 0 ]; then
        ((passes++))
    else
        ((fails++))
    fi
    
done

# Exit as a pass or fail (for CI purposes)
if [ $fails -eq 0 ]; then
    echo "${GREEN}Success! All ${passes} test(s) passed!"
    exit 0
else
    total=$((passes + fails))
    echo "${RED}Failure! Only ${passes} test(s) passed out of ${total}."
    exit 1
fi