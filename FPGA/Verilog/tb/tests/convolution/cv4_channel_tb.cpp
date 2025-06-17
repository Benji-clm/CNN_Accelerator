#include "../base_testbench.h" // Assuming a common base testbench header
#include <cmath>               // For floating point comparisons
#include <Imath/half.h>        // For accurate FP16 conversion
#include <vector>              // For 2D image representation
#include <array>               // For column data
#include <iomanip>             // For formatted output

// --- Forward Declarations ---
class Vdut; // The Verilated DUT class name
class VerilatedVcdC;

// --- Global Testbench Variables ---
Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

using Imath::half;

// --- FP16 Conversion Helpers ---
// Converts a single-precision float to its 16-bit FP16 representation.
uint16_t float_to_fp16(float value) {
    half h(value);
    return h.bits();
}

// Converts a 16-bit FP16 value back to a single-precision float.
float fp16_to_float(uint16_t value) {
    half h;
    h.setBits(value);
    return (float)h;
}

// --- Testbench Class ---

class ChannelTestbench : public BaseTestbench {
protected:
    // Constants matching the Verilog module parameters
    static constexpr int IMG_SIZE = 5;
    static constexpr int KERNEL_SIZE = 4;
    static constexpr int NUM_CHANNELS = 8; // Matches INPUT_CHANNEL_NUMBER
    static constexpr int OUT_SIZE = IMG_SIZE - KERNEL_SIZE + 1; // 2

    // The bias must match the hardcoded 'localparam BIAS' in the Verilog DUT.
    static constexpr float BIAS = 0.308882;

    // Type definitions for clarity
    using Image = std::vector<std::vector<float>>;
    using Kernel = std::array<std::array<float, KERNEL_SIZE>, KERNEL_SIZE>;
    using ImageColumn = std::array<uint16_t, IMG_SIZE>;
    using KernelColumn = std::array<uint16_t, KERNEL_SIZE>;

    // Buffer to store the output from the DUT
    Image dut_output;
    int output_col_idx = 0;


    void initializeInputs() override {
        top->clk = 0;
        top->rst = 1;
        top->kernel_load = 0;
        top->valid_in = 0;
        // Initialize all data inputs to zero using unpacked array interfaces
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            for (int i = 0; i < IMG_SIZE; ++i) {
                top->input_columns[ch][i] = 0;
            }
            for (int i = 0; i < KERNEL_SIZE; ++i) {
                top->kernel_inputs[ch][i] = 0;
            }
        }
    }

    // Helper method to cycle the clock and dump waveform data
    void clockTick(int n = 1) {
        for (int i = 0; i < n; i++) {
            ticks++;
            top->clk = 1;
            top->eval();
            tfp->dump(ticks);
            if (top->valid_out) {
                captureOutput();
            }
            ticks++;
            top->clk = 0;
            top->eval();
            tfp->dump(ticks);
        }
    }

    // Helper to load all kernels into the channel module
    void loadAllKernels(const std::vector<Kernel>& kernels) {
        top->kernel_load = 1;
        top->valid_in = 1;

        for (int col = 0; col < KERNEL_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                // Drive each kernel column for each channel
                for (int row = 0; row < KERNEL_SIZE; ++row) {
                    top->kernel_inputs[ch][row] = float_to_fp16(kernels[ch][row][col]);
                }
            }
            clockTick();
        }

        top->kernel_load = 0;
        top->valid_in = 0;
        clockTick(); // Ensure control signals are registered
    }

    // --- Input Driving Helper ---
    // Drives a single column of image data for each channel
    void driveInputColumns(const std::vector<ImageColumn>& columns) {
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            for (int row = 0; row < IMG_SIZE; ++row) {
                top->input_columns[ch][row] = columns[ch][row];
            }
        }
    }
    
    // --- Output Capture and Buffer Management ---
    void resetOutputBuffer() {
        dut_output.assign(OUT_SIZE, std::vector<float>(OUT_SIZE, 0.0f));
        output_col_idx = 0;
    }

    void captureOutput() {
        if (output_col_idx < OUT_SIZE) {
            // Read directly from the unpacked output array
            for (int i = 0; i < OUT_SIZE; i++) {
                dut_output[i][output_col_idx] = fp16_to_float(top->output_column[i]);
            }
            output_col_idx++;
        }
    }

    void convolve_2d(const Image& input, const Kernel& kernel, Image& output) {
        for (int y = 0; y < OUT_SIZE; y++) {
            for (int x = 0; x < OUT_SIZE; x++) {
                float sum = 0.0f;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        sum += input[y + ky][x + kx] * kernel[ky][kx];
                    }
                }
                output[y][x] = sum;
            }
        }
    }
};

// --- Test Case ---
TEST_F(ChannelTestbench, FullImageProcessingTest) {
    // 1. Reset the DUT
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    std::cout << "DUT Reset." << std::endl;

    // 2. Define Kernels and Input Data
    std::vector<Kernel> kernels = {
        // Input Channel 1
        {{
            {0.416782f, 0.289301f, 0.287646f, -0.194643f},
            {0.127622f, 0.034567f, 0.087225f, -0.359936f},
            {-0.018299f, 0.054776f, -0.064013f, -0.560707f},
            {-0.274767f, -0.270076f, -0.260784f, -0.390940f}
        }},
        // Input Channel 2
        {{
            {-0.438635f, -0.207532f, -0.048958f, 0.170026f},
            {0.178968f, -0.018574f, 0.038664f, 0.175771f},
            {0.427600f, 0.140888f, -0.136248f, -0.140984f},
            {-0.044425f, -0.093143f, -0.426275f, 0.112573f}
        }},
        // Input Channel 3
        {{
            {-0.030786f, -0.283789f, -0.460596f, -0.192574f},
            {-0.085704f, -0.280443f, 0.313264f, 0.315388f},
            {0.183992f, -0.825927f, -0.008822f, 0.058751f},
            {-0.100503f, -0.221761f, -0.075334f, -0.209413f}
        }},
        // Input Channel 4
        {{
            {-0.498581f, -0.029577f, 0.002176f, 0.085131f},
            {-0.815562f, 0.099853f, -0.082564f, -0.000717f},
            {-0.096059f, 0.129749f, -0.044991f, -0.334295f},
            {0.043618f, 0.081390f, 0.049328f, -0.351628f}
        }},
        // Input Channel 5
        {{
            {-0.498478f, -0.239227f, 0.031931f, -0.311656f},
            {-0.323574f, 0.061517f, 0.014162f, -0.243260f},
            {-0.227077f, 0.244750f, 0.240275f, -0.351388f},
            {0.147801f, 0.117028f, -0.136556f, -0.612970f}
        }},
        // Input Channel 6
        {{
            {-0.175232f, -0.621313f, -0.467735f, -0.187767f},
            {0.305555f, 0.118654f, -0.015520f, -0.157755f},
            {0.122083f, 0.185207f, 0.096220f, 0.047314f},
            {-0.105237f, 0.124150f, 0.193129f, 0.571917f}
        }},
        // Input Channel 7
        {{
            {-0.015368f, 0.143646f, 0.211270f, 0.387525f},
            {-0.312104f, -0.058727f, -0.145188f, 0.044629f},
            {-0.887316f, -0.330498f, -0.041483f, 0.031154f},
            {0.017059f, -0.218214f, -0.476907f, -0.055862f}
        }},
        // Input Channel 8
        {{
            {0.118733f, 0.160489f, -0.028195f, 0.028167f},
            {-0.056455f, -0.049376f, -0.089536f, -0.090609f},
            {-0.345509f, -0.177780f, -0.095449f, -0.093892f},
            {-0.035643f, 0.017947f, 0.293603f, 0.164338f}
        }}
    };

    std::vector<Image> inputs(NUM_CHANNELS, Image(IMG_SIZE, std::vector<float>(IMG_SIZE)));
    // Create simple ramp images for testing
    for(int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            inputs[0][i][j] = static_cast<float>(i * 0.2f + j * 0.4f);
            inputs[1][i][j] = 0.5f;
            inputs[2][i][j] = static_cast<float>(j * i * 0.1f);
            inputs[3][i][j] = static_cast<float>(i);
            inputs[4][i][j] = static_cast<float>(i * 0.2f + j * 0.4f);
            inputs[5][i][j] = static_cast<float>(j);
            // CORRECTED: Replaced division by 'i' with a safe alternative
            inputs[6][i][j] = static_cast<float>(i * 0.3f + j * 0.3f); 
            inputs[7][i][j] = static_cast<float>(i * i + j * j * j);
        }
    }

    // 3. Load all kernels into the DUT
    loadAllKernels(kernels);
    std::cout << "Kernels loaded into DUT." << std::endl;

    // 4. Calculate Golden Reference Result
    Image golden_output(OUT_SIZE, std::vector<float>(OUT_SIZE, 0.0f));
    std::vector<Image> filter_outputs(NUM_CHANNELS, Image(OUT_SIZE, std::vector<float>(OUT_SIZE)));
    
    for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
        convolve_2d(inputs[ch], kernels[ch], filter_outputs[ch]);
    }
    
    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            float total_sum = 0.0f;
            for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
                total_sum += filter_outputs[ch][y][x];
            }
            golden_output[y][x] = total_sum + BIAS;
        }
    }
    std::cout << "Reference output calculated." << std::endl;

    // 5. Stream all input columns into the DUT
    std::cout << "Streaming " << IMG_SIZE << " columns into DUT..." << std::endl;
    resetOutputBuffer();
    for (int col = 0; col < IMG_SIZE; ++col) {
        std::vector<ImageColumn> all_columns(NUM_CHANNELS);
        for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
            for(int row = 0; row < IMG_SIZE; ++row) {
                all_columns[ch][row] = float_to_fp16(inputs[ch][row][col]);
            }
        }
        
        driveInputColumns(all_columns);

        top->valid_in = 1;
        clockTick();
        top->valid_in = 0;
        
        // This delay can be adjusted based on the internal pipeline depth of the DUT
        clockTick(4);
    }
    
    // Ensure the last columns are processed
    clockTick(5); // Add a few extra ticks for safety
    std::cout << "Finished streaming. " << output_col_idx << " output columns captured." << std::endl;

    // 6. Verify the DUT's output against the golden model
    ASSERT_EQ(output_col_idx, OUT_SIZE) << "Incorrect number of output columns were captured.";
    std::cout << "Verifying output image..." << std::endl;
    int mismatches = 0;
    const float absolute_tolerance = 1e-5;
    const float relative_tolerance = 0.01f;

    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            float actual = dut_output[y][x];
            float expected = golden_output[y][x];
            
            // CORRECTED: Use a robust check for floating point comparison
            if (std::abs(actual - expected) > absolute_tolerance + relative_tolerance * std::abs(expected)) {
                if (mismatches < 10) { // Print first 10 mismatches
                    std::cerr << "Mismatch at (" << y << ", " << x << "): "
                              << "Expected=" << expected << ", Got=" << actual << std::endl;
                }
                mismatches++;
            }
        }
    }
    EXPECT_EQ(mismatches, 0) << "Found " << mismatches << " mismatches in the output image.";
    if (mismatches == 0) {
        std::cout << "SUCCESS: DUT output matches the reference." << std::endl;
    }
}

// --- Main Function ---
int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("channel_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    top->final();
    tfp->close();
    delete top;
    delete tfp;

    return res;
}
