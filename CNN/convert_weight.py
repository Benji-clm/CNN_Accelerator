import numpy as np

def float_to_fp16_hex(value):
    """Convert a float to its 16-bit FP16 hexadecimal representation."""
    fp16 = np.float16(value)
    uint16 = fp16.view(np.uint16)
    return f"16'h{uint16:04X}"

def parse_input_file(input_filename):
    """Parse the input file and return kernels and biases as lists."""
    kernels = []  # [filter][channel][values]
    biases = []
    
    with open(input_filename, 'r') as file:
        lines = [line.strip() for line in file.readlines() if line.strip()]
        i = 0
        
        # Parse kernel values
        while i < len(lines) and not lines[i].startswith("Layer: conv3.bias"):
            line = lines[i]
            if line.startswith("Filter"):
                filter_num = int(line.split()[1][:-1]) - 1
                kernels.append([])
                i += 1
                
                while i < len(lines) and lines[i].startswith("Input Channel"):
                    channel_num = int(lines[i].split()[2][:-1]) - 1
                    kernels[filter_num].append([])
                    i += 1
                    
                    # Read the next 4 lines for the 4x4 kernel
                    for _ in range(4):
                        kernel_line = lines[i]
                        values = [float(v) for v in kernel_line.split()]
                        kernels[filter_num][channel_num].extend(values)
                        i += 1
            else:
                i += 1
        
        # Parse bias values
        while i < len(lines) and not lines[i].startswith("Values:"):
            i += 1
        i += 1  # Skip "Values:" line
        
        while i < len(lines) and lines[i].startswith("Bias for Filter"):
            bias_line = lines[i]
            bias_value = float(bias_line.split(":")[1].strip())
            biases.append(bias_value)
            i += 1
    
    return kernels, biases

def generate_sv_code(kernels, biases):
    """Generate SystemVerilog code from kernels and biases."""
    # Generate KERNELS array
    kernel_str = "localparam [15:0] KERNELS[0:9][0:7][0:15] = '{\n"
    for filter_idx, filter in enumerate(kernels):
        kernel_str += f"    // Filter {filter_idx}\n    '{{\n"
        for channel_idx, channel in enumerate(filter):
            hex_values = [float_to_fp16_hex(val) for val in channel]
            channel_str = "{" + ", ".join(hex_values) + "}"
            if channel_idx < len(filter) - 1:
                channel_str += ","
            kernel_str += f"        {channel_str}\n"
        kernel_str += "    }"
        if filter_idx < len(kernels) - 1:
            kernel_str += ","
        kernel_str += "\n"
    kernel_str += "};"
    
    # Generate BIASES array
    hex_biases = [float_to_fp16_hex(bias) for bias in biases]
    bias_str = "localparam [15:0] BIASES[0:9] = '{" + ", ".join(hex_biases) + "};"
    
    return kernel_str, bias_str

def write_output_file(output_filename, kernel_str, bias_str):
    """Write the SystemVerilog code to an output file."""
    with open(output_filename, 'w') as out_file:
        out_file.write("// Convolutional Layer Parameters\n\n")
        out_file.write(kernel_str + "\n\n")
        out_file.write(bias_str + "\n")

def main():
    input_filename = "conv_layer_2.txt"
    output_filename = "output.sv"
    
    # Parse the input file
    kernels, biases = parse_input_file(input_filename)
    
    # Verify the structure
    assert len(kernels) == 10, "Expected 10 filters"
    for f in kernels:
        assert len(f) == 8, "Each filter should have 8 channels"
        for c in f:
            assert len(c) == 16, "Each channel should have 16 values"
    assert len(biases) == 10, "Expected 10 biases"
    
    # Generate SystemVerilog code
    kernel_str, bias_str = generate_sv_code(kernels, biases)
    
    # Write to output file
    write_output_file(output_filename, kernel_str, bias_str)
    print(f"SystemVerilog code has been written to {output_filename}")

if __name__ == "__main__":
    main()