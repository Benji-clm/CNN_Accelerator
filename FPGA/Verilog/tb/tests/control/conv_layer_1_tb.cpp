#include "../base_testbench.h" // Assuming a common base testbench header
#include <cmath>                 // For math functions like round
#include <vector>                // For 2D image representation
#include <array>                 // For column data
#include <iomanip>               // For formatted output
#include <cstdint>               // For fixed-width integers like int16_t
#include <algorithm>             // For std::max/min

// --- Forward Declarations ---
class Vdut;
class VerilatedVcdC;

// --- Global Testbench Variables ---
Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

// --- Helper Functions ---

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

// --- Base Testbench Class ---

class ConvLayerTestbench : public BaseTestbench {
protected:
    // Constants matching the Verilog module parameters
    static constexpr int DATA_WIDTH = 16;
    static constexpr int IMG_SIZE = 12;
    static constexpr int KERNEL_SIZE = 3;
    static constexpr int NUM_CHANNELS = 8;
    static constexpr int NUM_INPUT_CHANNELS = 4;
    static constexpr int OUT_SIZE = (IMG_SIZE - KERNEL_SIZE + 1)/2;

    static constexpr float BIAS_FLOAT = -0.137897;

    // Type definitions for clarity using fixed-point representation
    using ImageFloat = std::vector<std::vector<float>>;
    using ImageQ14 = std::vector<std::vector<int16_t>>;
    using KernelQ14 = std::array<std::array<int16_t, KERNEL_SIZE>, KERNEL_SIZE>;
    using ImageColumnQ14 = std::array<int16_t, IMG_SIZE>;
    using OutputFeatureMaps = std::vector<ImageQ14>;

    OutputFeatureMaps dut_output_maps;
    OutputFeatureMaps dut_feature_maps;
    // Buffer to store the output from all 8 channels of the DUT
    int output_col_idx = 0;
    int FM_col_idx = 0;

    void initializeInputs() override {
        top->valid_in = 0;
        
        // Create a column of zeros to reset inputs
        std::array<uint16_t, IMG_SIZE> zero_col;
        zero_col.fill(0);

        // Drive all input columns with zeros
        drive_input_columns(zero_col, zero_col, zero_col, zero_col);
    }

    // Helper method to cycle the clock and dump waveform data
    void clockTick(int n = 1) {
        for (int i = 0; i < n; i++) {
            ticks++;
            top->clk = 1;
            top->eval();
            if (top->valid_out) {
                captureAllOutputs();
            }
            if (top->column_valid_out){
                captureAllFM();
            }
            tfp->dump(ticks);

            ticks++;
            top->clk = 0;
            top->eval();
            tfp->dump(ticks);
        }
    }
    
    // --- Input Driving ---
    // Note: The original testbench had packed inputs. This module uses separate ports.
    // --- Input Driving ---
    void drive_input_columns(
        const std::array<uint16_t, IMG_SIZE>& col0,
        const std::array<uint16_t, IMG_SIZE>& col1,
        const std::array<uint16_t, IMG_SIZE>& col2,
        const std::array<uint16_t, IMG_SIZE>& col3
    ) {
        // Correctly copy element by element from std::array to the C-style array
        for (int i = 0; i < IMG_SIZE; ++i) {
            top->input_columns[0][i] = col0[i];
            top->input_columns[1][i] = col1[i];
            top->input_columns[2][i] = col2[i];
            top->input_columns[3][i] = col3[i];
        }
    }

    // --- Output Capture ---
    void resetOutputBuffer() {
        dut_output_maps.assign(NUM_CHANNELS, ImageQ14(OUT_SIZE, std::vector<int16_t>(OUT_SIZE, 0)));
        dut_feature_maps.assign(NUM_CHANNELS, ImageQ14(OUT_SIZE * 2, std::vector<int16_t>(OUT_SIZE * 2, 0)));
        output_col_idx = 0;
        FM_col_idx = 0;
    }

    void captureOutput(int ch, int col_index) {
        for (int i = 0; i < OUT_SIZE; i++) {
            dut_output_maps[ch][i][col_index] = top->output_columns[ch][i];
        }
    }

    void captureFM(int ch, int col_index){
        for (int i = 0; i < OUT_SIZE * 2; i++) {
            dut_feature_maps[ch][i][col_index] = top->fm_columns[ch][i];
        }
    }

    void captureAllOutputs() {
        if (output_col_idx < OUT_SIZE) {
            // Iterate over each of the 8 output channels
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                captureOutput(ch, output_col_idx);
            }
            output_col_idx++;
        }
    }

    void captureAllFM() {
        if (FM_col_idx < OUT_SIZE * 2) {
            // Iterate over each of the 8 output channels
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                captureFM(ch, FM_col_idx);
            }
            FM_col_idx++;
        }
    }

    // --- Utility to print a feature map ---
    void printFeatureMap(int channel_idx) {
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n--- Sample of Captured Output for Channel " << channel_idx << " ---" << std::endl;
        for(int y = 0; y < OUT_SIZE; ++y) {
            for (int x = 0; x < OUT_SIZE; ++x) {
                std::cout << std::setw(10) << q2_14_to_float(dut_output_maps[channel_idx][y][x]) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << "--------------------------------------" << std::endl;
        std::cout << "\n--- Sample of Captured FM for Channel " << channel_idx << " ---" << std::endl;
        for(int y = 0; y < OUT_SIZE * 2; ++y) {
            for (int x = 0; x < OUT_SIZE * 2; ++x) {
                std::cout << std::setw(10) << q2_14_to_float(dut_feature_maps[channel_idx][y][x]) << " ";
            }
            std::cout << std::endl;
        }
    }
};

// --- Test Case: Full Image Processing ---
TEST_F(ConvLayerTestbench, StreamImageAndCaptureOutput) {
    // 1. Reset the DUT. The internal FSM will automatically load kernels for 3 cycles.
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    std::cout << "DUT Reset." << std::endl;

    // Wait for the 3-cycle kernel load to finish
    std::cout << "Waiting for automatic kernel load (3 cycles)..." << std::endl;
    clockTick(3);

    // 2. Define Input Data (a simple ramp image)
    ImageFloat input_image(IMG_SIZE, std::vector<float>(IMG_SIZE));
    for(int i = 0; i < IMG_SIZE; i++) {
        for (int j = 0; j < IMG_SIZE; j++) {
            input_image[i][j] = static_cast<float>(i * 0.1f - j * 0.05f);
        }
    }

    // 3. Stream all input columns into the DUT
    std::cout << "Streaming " << IMG_SIZE << " columns into DUT..." << std::endl;
    resetOutputBuffer();
    for (int col = 0; col < IMG_SIZE; ++col) {
        std::array<uint16_t, IMG_SIZE> col_data;
        for(int row = 0; row < IMG_SIZE; ++row) {
            col_data[row] = static_cast<uint16_t>(float_to_q2_14(input_image[row][col]));
        }

        top->valid_in = 1;
        // Drive the same ramp data into all 4 input channels for this test
        drive_input_columns(col_data, col_data, col_data, col_data);        
        clockTick();
        top->valid_in = 0;
        clockTick(3);
    }
    top->valid_in = 0; // De-assert valid after last column
    
    // 4. Add extra clock ticks to flush the pipeline completely
    // The latency is dependent on the internal cv3_channel. A few extra cycles are safe.
    clockTick(10); 
    std::cout << "Finished streaming. " << output_col_idx << " output columns were captured." << std::endl;

    // 5. Print a sample of the captured output for visual verification
    ASSERT_EQ(output_col_idx, OUT_SIZE) << "Incorrect number of output columns were captured.";
    
    printFeatureMap(0); // Print a sample from the first channel's output
    printFeatureMap(7); // Print a sample from the last channel's output
}

// --- Main Function ---
int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("conv_layer_1.vcd");

    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    top->final();
    tfp->close();
    delete top;
    delete tfp;

    return res;
}