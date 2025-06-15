#include "../base_testbench.h"
#include <vector>
#include <cmath>
#include <Imath/half.h>
#include <iostream>
#include <iomanip>

// Forward declarations from Verilator
class Vdut;
class VerilatedVcdC;

// Global testbench variables
Vdut* top;
VerilatedVcdC* tfp;
unsigned int ticks = 0;

using Imath::half;

// --- Constants for Test Dimensions ---
constexpr int IMG_SIZE = 12;
constexpr int K_SIZE = 3;
constexpr int OUT_SIZE = IMG_SIZE - K_SIZE + 1; // 10

// Helper: Convert float to IEEE 754 half-precision (FP16) using Imath
uint16_t float_to_fp16(float value) {
    half h(value);
    return h.bits();
}

// Helper: Convert FP16 bit pattern to float using Imath
float fp16_to_float(uint16_t value) {
    half h;
    h.setBits(value);
    return (float)h;
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
        top->kernel_column_0 = 0;
        top->kernel_column_1 = 0;
        top->kernel_column_2 = 0;
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

    void loadKernel(float kernel[3][3]) {
        top->kernel_load = 1;
        top->valid_in = 1;
        
        // Load kernel column by column (3 cycles)
        for (int i = 0; i < 3; i++) {
            top->kernel_column_0 = float_to_fp16(kernel[0][i]);
            top->kernel_column_1 = float_to_fp16(kernel[1][i]);
            top->kernel_column_2 = float_to_fp16(kernel[2][i]);
            clockTick();
        }
        
        // De-assert control signals
        top->kernel_load = 0;
        top->valid_in = 0;
        top->kernel_column_0 = 0;
        top->kernel_column_1 = 0;
        top->kernel_column_2 = 0;
        clockTick();
    }
    void drive_wide_column(const std::array<uint16_t, 12>& column_data) {
        for (int i = 0; i < 6; ++i) {
            uint32_t low_word = column_data[i * 2];
            uint32_t high_word = column_data[i * 2 + 1];
            top->input_column[i] = (high_word << 16) | low_word;
        }
    }

    // Helper to perform convolution in software for verification
    void runSoftwareConvolution(
        const std::vector<std::array<uint16_t, IMG_SIZE>>& image, // <-- CHANGED TYPE
        const float kernel[K_SIZE][K_SIZE],
        std::vector<std::vector<float>>& result)
    {
        for (int y = 0; y < OUT_SIZE; y++) {
            for (int x = 0; x < OUT_SIZE; x++) {
                float sum = 0.0f;
                for (int ky = 0; ky < K_SIZE; ky++) {
                    for (int kx = 0; kx < K_SIZE; kx++) {
                        // Get the column (x+kx) and row (y+ky) from the input image
                        uint16_t image_fp16 = image[x + kx][y + ky];
                        // Convert back to float to perform the multiplication
                        sum += fp16_to_float(image_fp16) * kernel[ky][kx];
                    }
                }
                result[y][x] = sum;
            }
        }
    }

    // Helper to check if float values are close enough
    bool isClose(float a, float b, float tolerance = 0.000005f) {
        return std::abs(a - b) <= tolerance;
    }

    // Buffer to store the output from the DUT
    std::vector<std::vector<float>> dut_output;
    int output_col_idx = 0;

    void resetOutputBuffer() {
        dut_output.assign(OUT_SIZE, std::vector<float>(OUT_SIZE, 0.0f));
        output_col_idx = 0;
    }

    void captureOutput() {
        if (output_col_idx < OUT_SIZE) {
            std::cout << "\n[DEBUG] Capturing Output Column: " << output_col_idx << std::endl;
            std::cout << "---------------------------------------------" << std::endl;
    
            // The DUT provides 10 results packed into 5 x 32-bit words.
            // We loop 5 times to read the 5 words.
            for (int i = 0; i < OUT_SIZE / 2; i++) {
                // 1. Get the raw 32-bit data from the DUT's output port
                uint32_t raw_32bit_val = top->output_column[i];
    
                // 2. Unpack the lower and upper 16-bit (FP16) values
                uint16_t low_fp16  = raw_32bit_val & 0xFFFF;
                uint16_t high_fp16 = (raw_32bit_val >> 16) & 0xFFFF;
    
                // 3. Convert both to float and store them in our results buffer
                float low_float  = fp16_to_float(low_fp16);
                float high_float = fp16_to_float(high_fp16);
    
                // Check array bounds before assigning
                if ((i * 2) < OUT_SIZE) {
                    dut_output[i * 2][output_col_idx] = low_float;
                }
                if ((i * 2 + 1) < OUT_SIZE) {
                    dut_output[i * 2 + 1][output_col_idx] = high_float;
                }
    
                // 4. Print debug information for both unpacked values
                std::cout << "  output_column[" << std::setw(2) << std::setfill('0') << i * 2     << "]: "
                          << "FP16=0x" << std::hex << std::setw(4) << low_fp16 << std::dec
                          << ", Float=" << std::fixed << std::setprecision(4) << std::setw(10) << std::setfill(' ') << low_float
                          << std::endl;
                std::cout << "  output_column[" << std::setw(2) << std::setfill('0') << i * 2 + 1 << "]: "
                          << "FP16=0x" << std::hex << std::setw(4) << high_fp16 << std::dec
                          << ", Float=" << std::fixed << std::setprecision(4) << std::setw(10) << std::setfill(' ') << high_float
                          << std::endl;
            }
            std::cout << "---------------------------------------------" << std::endl;
    
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
    // The image is stored as a vector of columns, where each column is an array of 12 rows.
    // The data is pre-converted to uint16_t (FP16) format.
    std::vector<std::array<uint16_t, IMG_SIZE>> image(IMG_SIZE);

    for (int col = 0; col < IMG_SIZE; col++) {        // Iterate columns
        for (int row = 0; row < IMG_SIZE; row++) {    // Iterate rows
            // Create a simple gradient float value and convert it to FP16
            image[col][row] = float_to_fp16(static_cast<float>(row + col));
        }
    }

    // 2a. Display the initialized image for debugging
    std::cout << "\n--- Initialized Test Image (Converted to Float for Display) ---" << std::endl;
    // Header
    std::cout << "      ";
    for (int col = 0; col < IMG_SIZE; ++col) {
        std::cout << std::setw(7) << "Col " << col;
    }
    std::cout << std::endl;
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;
    // Body
    for (int row = 0; row < IMG_SIZE; ++row) {
        std::cout << "Row " << std::setw(2) << row << " |";
        for (int col = 0; col < IMG_SIZE; ++col) {
            // Convert back to float just for printing
            std::cout << std::fixed << std::setprecision(1) << std::setw(7) << fp16_to_float(image[col][row]);
        }
        std::cout << std::endl;
    }
    std::cout << "--------------------------------------------------------------------------------------------------" << std::endl;

    float kernel[K_SIZE][K_SIZE] = {
        {1.0f, 0.0f, -1.0f},
        {2.0f, 0.0f, -2.0f},
        {1.0f, 0.0f, -1.0f}
    }; // Sobel X edge detection

    // 3. Generate expected result using software model
    std::vector<std::vector<float>> expected_output(OUT_SIZE, std::vector<float>(OUT_SIZE));
    runSoftwareConvolution(image, kernel, expected_output);

    // 4. Load kernel into DUT
    loadKernel(kernel);

    // 5. Stream the entire image into the DUT, column by column
    for (int col = 0; col < IMG_SIZE; col++) {
        top->valid_in = 1;
        drive_wide_column(image[col]);
        clockTick();
        top->valid_in = 0;
        clockTick(2);
    }
    top->valid_in = 0;

    // 6. Add extra cycles to drain the pipeline of any remaining outputs
    clockTick(5);

    // 7. Verify the DUT's output against the software model
    for (int r = 0; r < OUT_SIZE; r++) {
        for (int c = 0; c < OUT_SIZE; c++) {
            EXPECT_TRUE(isClose(dut_output[r][c], expected_output[r][c]))
                << "Mismatch at output[" << r << "][" << c << "]: "
                << "Expected " << expected_output[r][c]
                << ", Got " << dut_output[r][c];
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