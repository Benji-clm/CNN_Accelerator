import numpy as np

def float_to_fp16_hex(value):
    fp16 = np.float16(value)
    # Convert to hex: get uint16 representation, then format as 4-digit hex string
    return format(np.frombuffer(fp16.tobytes(), dtype=np.uint16)[0], '04x')

def convert_file(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            try:
                float_val = float(line)
                hex_val = float_to_fp16_hex(float_val)
                outfile.write(f"{hex_val}\n")
            except ValueError:
                print(f"Fucking Bad Line: {line}")

if __name__ == "__main__":
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "kernel_floats.txt")
    output_path = os.path.join(script_dir, "kernel_fp16.mem")
    convert_file(input_path, output_path)
