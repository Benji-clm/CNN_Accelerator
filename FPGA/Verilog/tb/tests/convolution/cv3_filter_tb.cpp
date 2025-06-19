#include "../base_testbench.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm> // For std::max/min

// Forward declarations from Verilator
class Vdut;
class VerilatedVcdC;

// Global testbench variables
Vdut* top;
VerilatedVcdC* tfp;
unsigned int ticks = 0;

// --- Constants for Test Dimensions ---
constexpr int IMG_SIZE = 12;
constexpr int K_SIZE = 3;
constexpr int OUT_SIZE = IMG_SIZE - K_SIZE + 1; // 10

// --- Q2.14 Fixed-Point Conversion Helpers ---
// The Q2.14 format in a 16-bit signed integer means:
// - 1 bit for the sign
// - 1 bit for the integer part
// - 14 bits for the fractional part
// This gives a range of [-2.0, 1.9999...].
constexpr int Q_FRACTIONAL_BITS = 14;
constexpr float Q_SCALE_FACTOR = 1 << Q_FRACTIONAL_BITS; // 16384.0f

// Helper: Convert float to Q2.14 format (stored as int16_t)
int16_t float_to_q2_14(float value) {
    // Scale the float by 2^14
    float scaled_value = value * Q_SCALE_FACTOR;
    // Round to the nearest integer
    int32_t rounded_value = static_cast<int32_t>(round(scaled_value));
    // Saturate the value to the 16-bit signed integer range
    rounded_value = std::max(-32768, std::min(32767, rounded_value));
    return static_cast<int16_t>(rounded_value);
}

// Helper: Convert Q2.14 bit pattern (int16_t) to float for debugging/display
float q2_14_to_float(int16_t value) {
    return static_cast<float>(value) / Q_SCALE_FACTOR;
}


// Main testbench class derived from a base class
class ConvLayerTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->kernel_load = 0;
        top->valid_in = 0;
        for (int i = 0; i < IMG_SIZE; i++) {
            top->input_column[i] = 0;
        }
        for (int i = 0; i < K_SIZE; i++) {
            top->kernel_column[i] = 0;
        }
    }

    // Helper method to cycle the clock and dump waveform
    void clockTick(int n = 1) {
        for (int i = 0; i < n; i++) {
            // --- Half-cycle 1: Posedge ---
            ticks++;
            top->clk = 1;
            top->eval();    // Evaluate all sequential logic at posedge
            tfp->dump(ticks);
            if (top->valid_out) {
                captureOutput();
            }
            // --- Half-cycle 2: Negedge ---
            ticks++;
            top->clk = 0;
            top->eval();    // Evaluate any combinational logic changes
            tfp->dump(ticks);
            
        }
    }

    void loadKernel(float kernel[K_SIZE][K_SIZE]) {
        top->kernel_load = 1;
        top->valid_in = 1;
    
        // Load kernel column by column (K_SIZE cycles)
        for (int col = 0; col < K_SIZE; col++) {
            // Drive the unpacked kernel_column array
            for (int row = 0; row < K_SIZE; row++) {
                // Convert float kernel value to Q2.14 before driving the port
                top->kernel_column[row] = float_to_q2_14(kernel[row][col]);
            }
            clockTick();
        }
    
        // De-assert control signals
        top->kernel_load = 0;
        top->valid_in = 0;
        for (int row = 0; row < K_SIZE; row++) {
            top->kernel_column[row] = 0;
        }
        clockTick();
    }

    void driveInputColumn(const std::array<int16_t, IMG_SIZE>& column_data) {
        for (int i = 0; i < IMG_SIZE; ++i) {
            top->input_column[i] = column_data[i];
        }
    }

    // Helper to perform convolution in software using fixed-point arithmetic
    // to precisely model the hardware's behavior for verification.
    void runSoftwareConvolution(
        const std::vector<std::array<int16_t, IMG_SIZE>>& image_q,
        const int16_t kernel_q[K_SIZE][K_SIZE],
        std::vector<std::vector<int16_t>>& result_q)
    {
        for (int y = 0; y < OUT_SIZE; y++) {
            for (int x = 0; x < OUT_SIZE; x++) {
                // Use a 64-bit accumulator to avoid overflow during summation of products
                int64_t sum_accumulator = 0;
                for (int ky = 0; ky < K_SIZE; ky++) {
                    for (int kx = 0; kx < K_SIZE; kx++) {
                        int16_t image_val = image_q[x + kx][y + ky];
                        int16_t kernel_val = kernel_q[ky][kx];
                        // The product of two Q2.14 numbers results in a Q4.28 number.
                        // We cast to 32 bits before multiplication.
                        sum_accumulator += static_cast<int32_t>(image_val) * static_cast<int32_t>(kernel_val);
                    }
                }
                // The sum is in a Qx.28 format. To convert back to Q2.14, we must shift
                // right by the number of fractional bits (14).
                // To implement rounding instead of truncation, we add half of the divisor before shifting.
                int64_t rounding_val = (1 << (Q_FRACTIONAL_BITS - 1));
                sum_accumulator += rounding_val;
                int32_t shifted_sum = static_cast<int32_t>(sum_accumulator >> Q_FRACTIONAL_BITS);

                // Saturate the result to fit into a 16-bit signed integer
                shifted_sum = std::max(-32768, std::min(32767, shifted_sum));

                result_q[y][x] = static_cast<int16_t>(shifted_sum);
            }
        }
    }

    // Buffer to store the output from the DUT
    std::vector<std::vector<int16_t>> dut_output;
    int output_col_idx = 0;

    void resetOutputBuffer() {
        dut_output.assign(OUT_SIZE, std::vector<int16_t>(OUT_SIZE, 0));
        output_col_idx = 0;
    }

    void captureOutput() {
        if (output_col_idx < OUT_SIZE) {
            std::cout << "\n[DEBUG] Capturing Output Column: " << output_col_idx << std::endl;
            std::cout << "--------------------------------------------------------" << std::endl;
        
            // Loop through all elements of the unpacked output array
            for (int i = 0; i < OUT_SIZE; i++) {
                // 1. Get the raw 16-bit data from the DUT's output port
                //    It's treated as a signed int16_t for our Q2.14 format.
                int16_t raw_q14 = top->output_column[i];
            
                // 2. Store in our results buffer
                dut_output[i][output_col_idx] = raw_q14;
            
                // 3. Print debug information
                std::cout << "   output_column[" << std::setw(2) << std::setfill('0') << i << "]: "
                          << "Q14_val=" << std::setw(6) << std::setfill(' ') << raw_q14
                          << ", Float=" << std::fixed << std::setprecision(4) << std::setw(10) << std::setfill(' ') << q2_14_to_float(raw_q14)
                          << std::endl;
            }
            std::cout << "--------------------------------------------------------" << std::endl;
        
            output_col_idx++;
        }
    }
};

// The main test case for a full image
TEST_F(ConvLayerTestbench, FullImageProcessingTest)
{
    // 1. Reset the DUT
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    resetOutputBuffer();

    // 2. Define test image and kernel
    // The image is stored as a vector of columns.
    // The data is converted to Q2.14 format (int16_t).
    std::vector<std::array<int16_t, IMG_SIZE>> image(IMG_SIZE);
    for (int col = 0; col < IMG_SIZE; col++) {
        for (int row = 0; row < IMG_SIZE; row++) {
            // Create a simple gradient float value and convert it to Q2.14
            float val = static_cast<float>(row * 0.1f + col * 0.05f - 0.5f);
            image[col][row] = float_to_q2_14(val);
        }
    }
    
    // Kernel with both positive and negative values
    float kernel[K_SIZE][K_SIZE] = {
        {0.1146,  0.2144,   0.0849},
        {0.2259,  0.2235,   0.0743},
        {0.5138, -0.1771,  -0.0532}
    };
    // Convert float kernel to Q2.14 for the software model
    int16_t kernel_q[K_SIZE][K_SIZE];
    for (int r = 0; r < K_SIZE; r++) {
        for (int c = 0; c < K_SIZE; c++) {
            kernel_q[r][c] = float_to_q2_14(kernel[r][c]);
        }
    }


    // 3. Generate expected result using the fixed-point software model
    std::vector<std::vector<int16_t>> expected_output(OUT_SIZE, std::vector<int16_t>(OUT_SIZE));
    runSoftwareConvolution(image, kernel_q, expected_output);

    // 4. Load kernel into DUT (pass the float version, it converts inside)
    loadKernel(kernel);

    // 5. Stream the entire image into the DUT, column by column
    
    for (int col = 0; col < IMG_SIZE; col++) {
        top->valid_in = 1;
        driveInputColumn(image[col]);
        clockTick();
        top->valid_in = 0;
        clockTick(3);
    }
    top->valid_in = 0;

    // 6. Add extra cycles to drain the pipeline of any remaining outputs
    clockTick(5);

    // 7. Verify the DUT's output against the software model's raw Q2.14 output
    for (int r = 0; r < OUT_SIZE; r++) {
        for (int c = 0; c < OUT_SIZE; c++) {
            float expected_float = q2_14_to_float(expected_output[r][c]);
            float actual_float = q2_14_to_float(dut_output[r][c]);
            EXPECT_NEAR(actual_float, expected_float, 2.0f / Q_SCALE_FACTOR)
                << "Mismatch at output[" << r << "][" << c << "]: "
                << "Expected " << expected_output[r][c]
                << " (" << q2_14_to_float(expected_output[r][c]) << "), Got "
                << dut_output[r][c] << " (" << q2_14_to_float(dut_output[r][c]) << ")";
        }
    }
}

int main(int argc, char **argv)
{
    Verilated::commandArgs(argc, argv);
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("cv3_filter_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    top->final();
    tfp->close();
    delete top;
    delete tfp;

    return res;
}