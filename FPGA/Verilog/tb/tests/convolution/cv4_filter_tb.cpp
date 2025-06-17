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
constexpr int IMG_SIZE = 5;
constexpr int K_SIZE = 4;
constexpr int OUT_SIZE = IMG_SIZE - K_SIZE + 1; // 2

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
                top->kernel_column[row] = float_to_fp16(kernel[row][col]);
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

    void driveInputColumn(const std::array<uint16_t, IMG_SIZE>& column_data) {
        for (int i = 0; i < IMG_SIZE; ++i) {
            top->input_column[i] = column_data[i];
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
                result[y][x] = fp16_to_float(float_to_fp16(sum));
            }
        }
    }

    // Helper to check if float values are close enough
    bool isClose(float a, float b, float tolerance = 0.005f) {
        return std::abs(a - b) / b <= tolerance;
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
    
            // Loop through all 10 elements of the unpacked output array
            for (int i = 0; i < OUT_SIZE; i++) {
                // 1. Get the raw 16-bit data from the DUT's output port
                uint16_t raw_fp16 = top->output_column[i];
    
                // 2. Convert to float and store in our results buffer
                float float_val = fp16_to_float(raw_fp16);
                dut_output[i][output_col_idx] = float_val;
    
                // 3. Print debug information
                std::cout << "   output_column[" << std::setw(2) << std::setfill('0') << i << "]: "
                          << "FP16=0x" << std::hex << std::setw(4) << raw_fp16 << std::dec
                          << ", Float=" << std::fixed << std::setprecision(4) << std::setw(10) << std::setfill(' ') << float_val
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
            image[col][row] = float_to_fp16(static_cast<float>(row * 0.3f + col * 0.1f));
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
        {-0.498581, -0.029577, 0.002176, 0.085131},
        {-0.815562, 0.099853, -0.082564, -0.000717},
        {-0.096059, 0.129749, -0.044991, -0.334295},
        {0.043618, 0.081390, 0.049328, -0.351628}
    };

    // 3. Generate expected result using software model
    std::vector<std::vector<float>> expected_output(OUT_SIZE, std::vector<float>(OUT_SIZE));
    runSoftwareConvolution(image, kernel, expected_output);

    // 4. Load kernel into DUT
    loadKernel(kernel);

    // 5. Stream the entire image into the DUT, column by column
    for (int col = 0; col < IMG_SIZE; col++) {
        top->valid_in = 1;
        driveInputColumn(image[col]);
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