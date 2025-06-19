#include "../base_testbench.h" // Assuming a common base testbench header
#include <cmath>                 // For math functions like round
#include <vector>                // For 2D image representation
#include <array>                 // For column data
#include <iomanip>               // For formatted output
#include <cstdint>               // For fixed-width integers like int16_t
#include <algorithm>             // For std::max/min

// --- Forward Declarations ---
class Vdut; // The Verilated DUT class name
class VerilatedVcdC;

// --- Global Testbench Variables ---
Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

// --- Q2.14 Fixed-Point Conversion Helpers ---
// The Q2.14 format in a 16-bit signed integer means:
// - 1 bit for the sign
// - 1 bit for the integer part
// - 14 bits for the fractional part
constexpr int Q_FRACTIONAL_BITS = 14;
constexpr float Q_SCALE_FACTOR = 1 << Q_FRACTIONAL_BITS; // 16384.0f

// Helper: Convert float to Q2.14 format (stored as int16_t)
int16_t float_to_q2_14(float value) {
    float scaled_value = value * Q_SCALE_FACTOR;
    int32_t rounded_value = static_cast<int32_t>(round(scaled_value));
    return static_cast<int16_t>(std::max(-32768, std::min(32767, rounded_value)));
}

// Helper: Convert Q2.14 bit pattern (int16_t) to float for debugging/display
float q2_14_to_float(int16_t value) {
    return static_cast<float>(value) / Q_SCALE_FACTOR;
}


// --- Testbench Class ---
class ChannelTestbench : public BaseTestbench {
protected:
    // Constants matching the Verilog module parameters
    static constexpr int IMG_SIZE = 12;
    static constexpr int KERNEL_SIZE = 3;
    static constexpr int NUM_CHANNELS = 4; // Matches INPUT_CHANNEL_NUMBER
    static constexpr int OUT_SIZE = IMG_SIZE - KERNEL_SIZE + 1; // 10

    // The bias must match the parameter in the Verilog DUT.
    // We define it as a float and convert it to Q2.14 where needed.
    static constexpr float BIAS_FLOAT = -0.137897;

    // Type definitions for clarity using fixed-point representation
    using ImageQ14 = std::vector<std::vector<int16_t>>;
    using KernelQ14 = std::array<std::array<int16_t, KERNEL_SIZE>, KERNEL_SIZE>;
    using ImageColumnQ14 = std::array<int16_t, IMG_SIZE>;

    // Buffer to store the output from the DUT
    ImageQ14 dut_output;
    int output_col_idx = 0;


    void initializeInputs() override {
        top->clk = 0;
        top->rst = 1;
        top->kernel_load = 0;
        top->valid_in = 0;
        // Initialize all data inputs to zero
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

    // Helper to load all kernels (as floats) into the channel module
    void loadAllKernels(const std::vector<std::array<std::array<float, KERNEL_SIZE>, KERNEL_SIZE>>& kernels) {
        top->kernel_load = 1;
        top->valid_in = 1;

        for (int col = 0; col < KERNEL_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                for (int row = 0; row < KERNEL_SIZE; ++row) {
                    // Convert float to Q2.14 before driving DUT port
                    top->kernel_inputs[ch][row] = float_to_q2_14(kernels[ch][row][col]);
                }
            }
            clockTick();
        }

        top->kernel_load = 0;
        top->valid_in = 0;
        clockTick(); // Ensure control signals are registered
    }

    // Drives a single column of image data for each channel
    void driveInputColumns(const std::vector<ImageColumnQ14>& columns) {
        for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
            for (int row = 0; row < IMG_SIZE; ++row) {
                top->input_columns[ch][row] = columns[ch][row];
            }
        }
    }
    
    // --- Output Capture and Buffer Management ---
    void resetOutputBuffer() {
        dut_output.assign(OUT_SIZE, std::vector<int16_t>(OUT_SIZE, 0));
        output_col_idx = 0;
    }

    void captureOutput() {
        if (output_col_idx < OUT_SIZE) {
            for (int i = 0; i < OUT_SIZE; i++) {
                // Store the raw Q2.14 value from the DUT
                dut_output[i][output_col_idx] = top->output_column[i];
            }
            output_col_idx++;
        }
    }

    // --- Bit-Accurate Golden Model for Verification ---
    void golden_convolve_q14(const ImageQ14& input, const KernelQ14& kernel, ImageQ14& output) {
        for (int y = 0; y < OUT_SIZE; y++) {
            for (int x = 0; x < OUT_SIZE; x++) {
                int64_t sum_accumulator = 0;
                for (int ky = 0; ky < KERNEL_SIZE; ky++) {
                    for (int kx = 0; kx < KERNEL_SIZE; kx++) {
                        int32_t image_val = input[y + ky][x + kx];
                        int32_t kernel_val = kernel[ky][kx];
                        sum_accumulator += image_val * kernel_val;
                    }
                }
                // Convert Qx.28 result back to Q2.14 with rounding
                int64_t rounded_sum = sum_accumulator + (1LL << (Q_FRACTIONAL_BITS - 1));
                int32_t shifted_sum = static_cast<int32_t>(rounded_sum >> Q_FRACTIONAL_BITS);
                
                // Saturate and store
                output[y][x] = static_cast<int16_t>(std::max(-32768, std::min(32767, shifted_sum)));
            }
        }
    }
};

// --- Test Case ---
TEST_F(ChannelTestbench, FullImageProcessingTest) {
    // 1. Reset the DUT
    top->rst = 1; clockTick(2); top->rst = 0; clockTick();
    std::cout << "DUT Reset." << std::endl;

    // 2. Define Kernels and Input Data as floats
    std::vector<std::array<std::array<float, KERNEL_SIZE>, KERNEL_SIZE>> kernels_f(NUM_CHANNELS);
    kernels_f[0] = {{{0.18, 0.09, -0.11}, {-0.42, 0.24, -0.02}, {0.38, -0.27, -0.04}}};
    kernels_f[1] = {{{0.05, 0.15, -0.18}, {-0.35, -0.37, 0.27}, {-0.44, 0.17, 0.31}}};
    kernels_f[2] = {{{0.23, 0.34, -0.00}, {0.03, -0.07, -0.18}, {-1.12, -0.62, -0.35}}};
    kernels_f[3] = {{{-0.30, 0.15, 0.27}, {0.14, 0.43, -0.05}, {-0.20, -0.33, 0.21}}};

    std::vector<std::vector<std::vector<float>>> inputs_f(NUM_CHANNELS, std::vector<std::vector<float>>(IMG_SIZE, std::vector<float>(IMG_SIZE)));
    for(int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            inputs_f[0][i][j] = (float)(i * 0.1 - j * 0.05);
            inputs_f[1][i][j] = 0.5f;
            inputs_f[2][i][j] = (float)(j * 0.1 - 0.5);
            inputs_f[3][i][j] = (float)(i * 0.1 - 0.5);
        }
    }
    
    // 3. Convert all test data to Q2.14 format for the golden model and DUT
    std::vector<KernelQ14> kernels_q(NUM_CHANNELS);
    for(int ch=0; ch<NUM_CHANNELS; ++ch) for(int r=0; r<KERNEL_SIZE; ++r) for(int c=0; c<KERNEL_SIZE; ++c)
        kernels_q[ch][r][c] = float_to_q2_14(kernels_f[ch][r][c]);

    std::vector<ImageQ14> inputs_q(NUM_CHANNELS, ImageQ14(IMG_SIZE, std::vector<int16_t>(IMG_SIZE)));
    for(int ch=0; ch<NUM_CHANNELS; ++ch) for(int r=0; r<IMG_SIZE; ++r) for(int c=0; c<IMG_SIZE; ++c)
        inputs_q[ch][r][c] = float_to_q2_14(inputs_f[ch][r][c]);

    const int16_t BIAS_Q14 = float_to_q2_14(BIAS_FLOAT);

    // 4. Calculate Golden Reference Result using bit-accurate Q2.14 model
    ImageQ14 golden_output(OUT_SIZE, std::vector<int16_t>(OUT_SIZE, 0));
    std::vector<ImageQ14> filter_outputs(NUM_CHANNELS, ImageQ14(OUT_SIZE, std::vector<int16_t>(OUT_SIZE)));
    
    for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
        golden_convolve_q14(inputs_q[ch], kernels_q[ch], filter_outputs[ch]);
    }
    
    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            int32_t total_sum = BIAS_Q14;
            for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
                total_sum += filter_outputs[ch][y][x];
            }
            golden_output[y][x] = static_cast<int16_t>(std::max(-32768, std::min(32767, total_sum)));
        }
    }
    std::cout << "Golden reference output calculated." << std::endl;

    // 5. Load kernels and stream input image into DUT
    loadAllKernels(kernels_f);
    std::cout << "Kernels loaded into DUT." << std::endl;

    std::cout << "Streaming " << IMG_SIZE << " columns into DUT..." << std::endl;
    resetOutputBuffer();
    
    for (int col = 0; col < IMG_SIZE; ++col) {
        std::vector<ImageColumnQ14> all_columns(NUM_CHANNELS);
        for(int ch = 0; ch < NUM_CHANNELS; ++ch) {
            for(int row = 0; row < IMG_SIZE; ++row) {
                all_columns[ch][row] = inputs_q[ch][row][col];
            }
        }
        top->valid_in = 1;
        driveInputColumns(all_columns);
        clockTick();
        top->valid_in = 0;
        clockTick(3);
    }
    top->valid_in = 0;
    clockTick(10); // Add extra ticks to drain the pipeline
    std::cout << "Finished streaming. " << output_col_idx << " output columns captured." << std::endl;


    // 6. Verify the DUT's output against the golden model
    ASSERT_EQ(output_col_idx, OUT_SIZE) << "Incorrect number of output columns were captured.";
    std::cout << "Verifying " << OUT_SIZE << "x" << OUT_SIZE << " output image..." << std::endl;
    for (int y = 0; y < OUT_SIZE; y++) {
        for (int x = 0; x < OUT_SIZE; x++) {
            // Compare raw fixed-point values for an exact match
            EXPECT_NEAR(q2_14_to_float(golden_output[y][x]), q2_14_to_float(dut_output[y][x]), 6.0f / Q_SCALE_FACTOR)
                << "Mismatch at (" << y << ", " << x << "): "
                << "Expected=" << golden_output[y][x] << " (" << q2_14_to_float(golden_output[y][x])
                << "), Got=" << dut_output[y][x] << " (" << q2_14_to_float(dut_output[y][x]) << ")";
        }
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
