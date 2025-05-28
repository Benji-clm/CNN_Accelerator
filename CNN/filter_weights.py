import struct

def float_to_fp16_hex(f):
    """
    Converts a single-precision float to a half-precision float (FP16)
    and returns its hexadecimal representation.
    """
    # Pack the float into a 4-byte binary string (single precision)
    float_bytes = struct.pack('>f', f) # Use big-endian for consistent byte order

    # Convert to half-precision (FP16)
    # This involves a bit of manual manipulation as Python's struct
    # doesn't directly support FP16 for packing/unpacking in older versions
    # and we need to handle the conversion logic.
    # For simplicity and accuracy, we'll use a common approach that
    # correctly converts the bit pattern.

    # Get the 32-bit integer representation of the float
    f_int = struct.unpack('>I', float_bytes)[0]

    sign = (f_int >> 31) & 0x01
    exponent = ((f_int >> 23) & 0xFF) - 127 # Adjust for FP32 bias
    mantissa = f_int & 0x7FFFFF

    if exponent == 128:  # Infinity or NaN
        if mantissa == 0:
            fp16_val = (sign << 15) | (0x1F << 10) # Infinity
        else:
            fp16_val = (sign << 15) | (0x1F << 10) | (mantissa >> 13) | 0x01 # NaN (set a bit in mantissa)
    elif exponent > 15: # Overflow, represent as infinity
        fp16_val = (sign << 15) | (0x1F << 10)
    elif exponent < -14: # Underflow, represent as zero or subnormal
        if exponent >= -24: # Subnormal
            # Denormalized number
            shifted_mantissa = (1 << 23) | mantissa
            fp16_val = (sign << 15) | (shifted_mantissa >> (-14 - exponent))
        else: # Too small to be represented, flush to zero
            fp16_val = (sign << 15)
    else: # Normal number
        fp16_val = (sign << 15) | ((exponent + 15) << 10) | (mantissa >> 13)

    return f"0x{fp16_val:04X}"


def process_filters(input_data):
    """
    Processes the input filter data, converts each value to FP16 hex,
    and prints the result.
    """
    filter_blocks = input_data.strip().split('Filter ')
    
    for block in filter_blocks:
        if not block:
            continue
        
        filter_num = block.split(':')[0].strip()
        print(f"Filter {filter_num}:")
        print("  Input Channel 1:")
        
        # Extract the lines containing numerical data
        lines = block.split('\n')
        data_lines = []
        for line in lines[1:]:
            stripped_line = line.strip()
            if stripped_line and (stripped_line[0].isdigit() or stripped_line[0] == '-' or stripped_line[0] == '.'):
                data_lines.append(stripped_line)

        for line in data_lines:
            values_str = line.split()
            fp16_values = [float_to_fp16_hex(float(val)) for val in values_str]
            print("   " + " ".join(fp16_values))

# Your provided input data
input_data = """
Filter 1:
  Input Channel 1:
    0.142142 -0.022964 0.051559 -0.040001 0.341916
    0.335964 0.293502 0.205089 0.360023 0.309603
    0.143716 0.202700 0.240360 0.178318 -0.123146
    -0.163367 -0.109109 -0.211772 -0.222878 -0.192655
    -0.315286 -0.170149 -0.269675 -0.059139 -0.101655
Filter 2:
  Input Channel 1:
    0.042353 -0.257297 -0.176743 0.235938 0.502111
    -0.364804 -0.227053 0.348920 0.353059 0.248846
    -0.316059 0.177530 0.632040 0.107104 -0.323250
    0.185381 0.299719 0.401262 -0.226486 -0.353403
    0.554716 0.336709 -0.149258 -0.220108 -0.113286
Filter 3:
  Input Channel 1:
    -0.010178 -0.162291 -0.145069 -0.258341 -0.071704
    -0.219291 -0.091044 -0.012137 0.042596 -0.131532
    0.227248 -0.087016 -0.151780 0.226600 0.197701
    0.294168 0.106725 -0.107722 0.028459 0.177983
    0.153855 -0.011665 -0.128706 -0.177332 -0.106230
Filter 4:
  Input Channel 1:
    -0.034530 0.005907 0.019674 0.014274 -0.175976
    -0.097438 -0.168381 -0.307375 -0.212568 -0.330135
    -0.010184 0.025795 -0.193534 -0.075352 -0.032359
    0.303746 0.254727 0.249843 0.206946 -0.064409
    0.055382 0.189840 0.068430 0.204641 0.064606
Filter 5:
  Input Channel 1:
    -0.044354 0.301579 0.087666 0.205082 0.149888
    -0.106403 0.260637 0.014473 -0.148101 -0.199880
    -0.040474 0.365513 0.094822 -0.172029 -0.285909
    0.050908 0.443437 0.030324 -0.176127 -0.218629
    0.280402 0.152458 -0.151974 -0.159492 -0.126708
Filter 6:
  Input Channel 1:
    -0.070961 0.159822 0.086177 -0.165139 -0.115891
    0.158516 0.234684 -0.193757 -0.151889 -0.230972
    0.273310 0.062809 -0.018456 -0.270638 0.031452
    0.175190 0.189229 -0.143243 -0.132037 0.053050
    0.107332 0.101716 0.133993 -0.095584 0.036609
Filter 7:
  Input Channel 1:
    0.431156 0.351456 0.269795 -0.037735 -0.243497
    0.036214 0.151021 -0.116838 -0.209350 -0.054402
    0.033243 -0.238620 -0.084314 0.007685 0.472435
    -0.210019 -0.343756 -0.051817 0.352338 0.340484
    0.007876 0.218630 0.442936 0.085638 -0.283277
Filter 8:
  Input Channel 1:
    0.158801 -0.107058 -0.033301 -0.190059 -0.061217
    -0.123023 -0.174955 -0.160615 -0.078389 0.179645
    -0.208319 -0.117278 -0.085516 -0.055573 0.142481
    -0.228680 0.030366 -0.106342 0.184505 0.219968
    -0.223749 0.054608 0.208724 0.115451 0.060081
Filter 9:
  Input Channel 1:
    -0.109109 -0.085467 -0.100666 0.066739 -0.063752
    0.041295 0.031552 0.082585 -0.078653 0.292874
    -0.216535 0.050128 -0.167641 0.192485 0.224310
    -0.234482 -0.240796 0.079782 0.166608 0.015768
    0.085648 -0.255249 -0.151474 0.053662 -0.017623
Filter 10:
  Input Channel 1:
    -0.244974 0.019830 0.225589 0.486286 0.221941
    -0.041195 -0.311052 0.165427 0.199982 0.263133
    -0.343226 -0.206547 0.067463 0.300879 0.100251
    -0.204529 -0.159839 0.181945 0.119272 0.319413
    -0.240736 -0.035155 0.102382 0.379350 0.434171
Filter 11:
  Input Channel 1:
    -0.394504 -0.262918 -0.279427 -0.104717 -0.223929
    0.068549 -0.158178 -0.258891 -0.081648 -0.070108
    -0.001945 0.007910 0.184145 -0.129893 0.024364
    -0.140641 0.090032 -0.031271 0.196495 0.103000
    0.041190 0.173543 0.274771 0.173081 0.246132
Filter 12:
  Input Channel 1:
    0.035284 0.026650 0.035492 0.220980 0.196763
    0.022830 -0.151214 -0.103423 -0.159106 0.352052
    0.083360 -0.125599 -0.294032 -0.120730 0.516415
    0.001434 -0.030258 -0.271334 -0.279539 0.499783
    0.008634 -0.086980 -0.262255 -0.111254 0.337167
Filter 13:
  Input Channel 1:
    -0.117631 -0.334885 -0.257514 0.220318 0.038191
    0.136093 -0.086529 0.179383 0.380954 0.053846
    0.480471 0.119599 -0.074185 -0.056830 0.140164
    -0.136911 -0.052867 -0.399463 -0.251382 -0.045694
    -0.168686 -0.102428 0.107861 0.041357 0.116528
Filter 14:
  Input Channel 1:
    0.680214 -0.096468 -0.466482 -0.430906 -0.267889
    0.600550 0.569561 0.227629 -0.524484 -0.463165
    -0.165395 0.489430 0.784538 0.372293 -0.045339
    -0.375451 -0.020456 0.433471 0.480515 0.140327
    -0.346824 -0.425938 -0.199167 0.259502 0.629680
Filter 15:
  Input Channel 1:
    -0.288533 -0.325536 -0.104783 -0.136387 0.037284
    -0.008766 0.345385 0.396729 0.273462 0.195003
    0.413226 0.187017 -0.229794 -0.247074 -0.005848
    -0.094768 -0.435073 -0.347539 0.105048 0.022673
    0.018369 0.146374 0.205499 0.281429 -0.040816
Filter 16:
  Input Channel 1:
    0.250344 0.025087 0.166442 0.141881 0.416647
    0.097092 0.317234 0.316610 0.311423 0.124995
    -0.302764 -0.172890 -0.286928 -0.177844 -0.378473
    -0.157071 -0.301753 -0.079426 -0.239960 -0.266137
    0.146785 0.109813 -0.238513 -0.116644 0.170961
"""
process_filters(input_data)