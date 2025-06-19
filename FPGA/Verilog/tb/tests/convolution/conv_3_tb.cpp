/*
*  Verifies the results of the convolution module, exits with a 0 on success.
*/

#include "../base_testbench.h"
#include <cmath>        // For floating point comparisons
#include <Imath/half.h> // For accurate FP16 conversion

// Forward declarations
class Vdut;
class VerilatedVcdC;

// Global testbench variables
Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

// --- Q2.14 Fixed-Point Conversion Helpers ---

const float Q_SCALE_FACTOR = 1 << 14; // 2^14 = 16384.0f

// Helper: Convert float to Q2.14 (S1.14) signed 16-bit integer
uint16_t float_to_q2_14(float value) {
    // The range of Q2.14 (S1.14) is [-2.0, 1.9999...]
    float clamped_value = std::max(-2.0f, std::min(value, 2.0f - 1.0f/Q_SCALE_FACTOR));
    
    // Scale, round to nearest integer, and cast
    int16_t fixed_point_val = static_cast<int16_t>(std::round(clamped_value * Q_SCALE_FACTOR));
    
    // Return the raw bits for Verilator
    return static_cast<uint16_t>(fixed_point_val);
}

// Helper: Convert Q2.14 signed 16-bit integer back to float
float q2_14_to_float(uint16_t value) {
    // Cast the raw bits to a signed integer to interpret the sign bit correctly
    int16_t signed_val = static_cast<int16_t>(value);
    
    // Divide by the scale factor to get the float representation
    return static_cast<float>(signed_val) / Q_SCALE_FACTOR;
}


class ConvolutionTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->data_in[0] = 0;
        top->data_in[1] = 0;
        top->data_in[2] = 0;
        top->kernel_load = 0;
        top->valid_in = 0;
        top->valid_out = 0;
    }

    // Helper method to cycle the clock and dump waveform
    void clockTick(int n = 1) {
        for (int i = 0; i < n; i++) {
            ticks++;
            top->clk = 1;
            top->eval();
            if (tfp) tfp->dump(ticks);

            ticks++;
            top->clk = 0;
            top->eval();
            if (tfp) tfp->dump(ticks);
        }
    }
    
    // Helper method to load a 3x3 kernel
    void loadKernel(float kernel[3][3]) {
        top->kernel_load = 1;
        top->valid_in = 1;
        
        // Load kernel column by column (3 cycles)
        for (int i = 0; i < 3; i++) {
            top->data_in[0] = float_to_q2_14(kernel[0][i]);
            top->data_in[1] = float_to_q2_14(kernel[1][i]);
            top->data_in[2] = float_to_q2_14(kernel[2][i]);
            clockTick();
        }
        
        // De-assert control signals
        top->kernel_load = 0;
        top->valid_in = 0;
        // Allow one cycle for control signals to settle before processing image
        clockTick(); 
    }
    
    // Helper method to process a 3x3 image patch
    void processImage(float image[3][3]) {
        top->valid_in = 1;
        
        // Stream image data column by column (3 cycles)
        for (int i = 0; i < 3; i++) {
            top->data_in[0] = float_to_q2_14(image[0][i]);
            top->data_in[1] = float_to_q2_14(image[1][i]);
            top->data_in[2] = float_to_q2_14(image[2][i]);
            clockTick();
        }
        
        // De-assert valid_in after the last data is sent
        top->valid_in = 0;
        top->valid_out = 1;
        clockTick();
        top->valid_out = 0;
        clockTick();

    }
};

TEST_F(ConvolutionTestbench, IdentityKernelTest)
{
    // 1. Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image (values must be within Q2.14 range [-2.0, 2.0))
    float kernel[3][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };
    float image[3][3] = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    };
    
    // 3. Load Kernel and Process Image
    loadKernel(kernel);
    processImage(image);

    // 5. Check the result. Expected output is the center pixel: 0.5
    float expected_float = 0.5f;
    float actual_float = q2_14_to_float(top->data_out);
    
    EXPECT_NEAR(actual_float, expected_float, 1.0f / Q_SCALE_FACTOR)
        << "Expected " << expected_float << " but got " << actual_float;
}

TEST_F(ConvolutionTestbench, EdgeDetectionKernelTest)
{
    // 1. Reset
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image
    float kernel[3][3] = {
        { 1.0f,  1.0f,  1.0f},
        { 0.0f,  0.0f,  0.0f},
        {-1.0f, -1.0f, -1.0f}
    };
    float image[3][3] = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    };
    
    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    // 5. Check result: Sum(top row) - Sum(bottom row)
    // (0.1+0.2+0.3) - (0.7+0.8+0.9) = 0.6 - 2.4 = -1.8
    float expected_float = -1.8f;
    float actual_float = q2_14_to_float(top->data_out);

    EXPECT_NEAR(actual_float, expected_float, 2.0f / Q_SCALE_FACTOR) // slightly larger tolerance for MAC
        << "Expected " << expected_float << " but got " << actual_float;
}

TEST_F(ConvolutionTestbench, BlurKernelTest)
{
    // 1. Reset
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image. Values chosen so the result doesn't overflow Q2.14 range.
    float kernel[3][3] = {
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f},
        {0.1f, 0.1f, 0.1f}
    };
    float image[3][3] = {
        {0.2f, 0.2f, 0.2f},
        {0.2f, 0.2f, 0.2f},
        {0.2f, 0.2f, 0.2f}
    };

    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    
    // 5. Check result: 9 * (0.1 * 0.2) = 9 * 0.02 = 0.18
    float expected_float = 0.18f;
    float actual_float = q2_14_to_float(top->data_out);

    EXPECT_NEAR(actual_float, expected_float, 4.0f / Q_SCALE_FACTOR)
        << "Expected " << expected_float << " but got " << actual_float;
}

TEST_F(ConvolutionTestbench, SobelXKernelTest)
{
    // 1. Reset
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image. The '2.0' in the kernel will be clamped to the max Q2.14 value.
    float kernel[3][3] = {
        { 1.0f,  0.0f, -1.0f},
        { 2.0f,  0.0f, -2.0f},
        { 1.0f,  0.0f, -1.0f}
    };
    float image[3][3] = {
        {0.1f, 0.2f, 0.3f},
        {0.4f, 0.5f, 0.6f},
        {0.7f, 0.8f, 0.9f}
    };

    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    // 5. Check result: (1*0.1 + 2*0.4 + 1*0.7) - (1*0.3 + 2*0.6 + 1*0.9)
    // (0.1 + 0.8 + 0.7) - (0.3 + 1.2 + 0.9) = 1.6 - 2.4 = -0.8
    float expected_float = -0.8f;
    float actual_float = q2_14_to_float(top->data_out);

    // Note: Clamping of kernel values (2.0 and -2.0) introduces small errors
    EXPECT_NEAR(actual_float, expected_float, 3.0f / Q_SCALE_FACTOR) 
        << "Expected " << expected_float << " but got " << actual_float;
}

int main(int argc, char **argv)
{
    // Initialize Verilator
    Verilated::commandArgs(argc, argv);
    top = new Vdut;
    tfp = new VerilatedVcdC;

    // Enable tracing
    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("convolution_waveform_q2_14.vcd");

    // Initialize Google Test
    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    // Cleanup
    if (tfp) tfp->close();
    top->final();
    delete top;
    delete tfp;

    return res;
}