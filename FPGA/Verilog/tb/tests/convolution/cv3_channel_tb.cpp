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
    static constexpr int IMG_SIZE = 12;
    static constexpr int KERNEL_SIZE = 3;
    static constexpr int NUM_CHANNELS = 4; // Matches INPUT_CHANNEL_NUMBER
    static constexpr int OUT_SIZE = IMG_SIZE - KERNEL_SIZE + 1; // 10

    // The bias must match the hardcoded 'localparam BIAS' in the Verilog DUT.
    static constexpr float BIAS = -0.137897;

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

    // --- Golden Model for 2D Convolution ---
    void golden_convolve_2d(const Image& input, const Kernel& kernel, Image& output) {
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
    std::vector<Kernel> kernels(NUM_CHANNELS);
    kernels[0] = {{{0.182497, 0.095799, -0.118198},{-0.425404, 0.249551, -0.020287}, {0.381072, -0.278986, -0.047726}}};      // Identity
    kernels[1] = {{{0.054801, 0.153301, -0.185463}, {-0.353260, -0.370310, 0.272392}, {-0.449251, 0.170076, 0.313901}}};      // Box blur / Summing
    kernels[2] = {{{0.236041, 0.343119, -0.007934}, {0.033060, -0.075088, -0.185990}, {-1.121445, -0.623998, -0.355788}}};  // Sobel X
    kernels[3] = {{{-0.304607, 0.156517, 0.276813}, {0.149517, 0.437039, -0.052157}, {-0.203192, -0.336657, 0.213920}}};    // Sobel Y

    std::vector<Image> inputs(NUM_CHANNELS, Image(IMG_SIZE, std::vector<float>(IMG_SIZE)));
    // Create simple ramp images for testing
    for(int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            inputs[0][i][j] = static_cast<float>(i * 0.2 + j * 0.4);   // Ramp
            inputs[1][i][j] = 0.5f;                                   // Constant
            inputs[2][i][j] = static_cast<float>(j * i * 0.1);         // Vertical lines
            inputs[3][i][j] = static_cast<float>(i);                   // Horizontal lines
        }
    }

    // 3. Load all kernels into the DUT
    loadAllKernels(kernels);
    std::cout << "Kernels loaded into DUT." << std::endl;

    // 4. Calculate Golden Reference Result
    Image golden_output(OUT_SIZE, std::vector<float>(OUT_SIZE, 0.0f));
    std::vector<Image> filter_outputs(NUM_CHANNELS, Image(OUT_SIZE, std::vector<float>(OUT_SIZE)));
    
    for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
        golden_convolve_2d(inputs[ch], kernels[ch], filter_outputs[ch]);
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
    std::cout << "Golden reference output calculated." << std::endl;

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
    std::cout << "Verifying 10x10 output image..." << std::endl;
    int mismatches = 0;
    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            float actual = dut_output[y][x];
            float expected = golden_output[y][x];
            // Use a relative tolerance for floating point comparisons
            if (std::abs(actual - expected) / std::abs(expected) > 0.01f) { 
                if (mismatches < 10) { // Print first 10 mismatches
                    std::cerr << "Mismatch at (" << y << ", " << x << "): "
                              << "Expected=" << expected << ", Got=" << actual << std::endl;
                }
                mismatches++;
            }
        }
    }
    EXPECT_TRUE(mismatches < 3) << "Found " << mismatches << " mismatches in the output image.";
    if (mismatches == 0) {
        std::cout << "SUCCESS: DUT output matches golden model." << std::endl;
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
