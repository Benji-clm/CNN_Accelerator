#!/bin/bash
# Test script for display_tiler

echo "Compiling display_tiler testbench..."

# Create obj_dir if it doesn't exist
mkdir -p obj_dir

# Compile with Verilator
verilator --cc --exe --build \
    --top-module tb_display_tiler \
    -Wall -Wno-UNUSED \
    tb/tb_display_tiler.sv \
    rtl/display_tiler.sv \
    --exe-dir obj_dir \
    2>&1 | tee obj_dir/compile.log

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Running simulation..."
    
    cd obj_dir
    ./Vtb_display_tiler
    
    if [ -f display_output.ppm ]; then
        echo "Success! PPM file generated."
        echo "You can view it with:"
        echo "  - Online: upload display_output.ppm to photopea.com"
        echo "  - Linux: eog display_output.ppm (or gimp)"
        echo "  - Convert to PNG: convert display_output.ppm display_output.png"
    fi
    
    cd ..
else
    echo "Compilation failed. Check obj_dir/compile.log"
fi
