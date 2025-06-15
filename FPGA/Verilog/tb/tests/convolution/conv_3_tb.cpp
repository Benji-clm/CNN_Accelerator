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

using Imath::half;

// Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern using Imath
uint16_t float_to_fp16(float value) {
    half h(value); // Implicitly convert float to half
    return h.bits(); // Return the 16-bit raw data
}

// Helper: Convert FP16 bit pattern to float using Imath
float fp16_to_float(uint16_t value) {
    half h;
    h.setBits(value); // Directly set the raw bits
    return (float)h;  // Cast back to float
}

class ConvolutionTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->data_in0 = 0;
        top->data_in1 = 0;
        top->data_in2 = 0;
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
            tfp->dump(ticks);

            ticks++;
            top->clk = 0;
            top->eval();
            tfp->dump(ticks);
        }
    }
    
    // Helper method to load a 3x3 kernel
    void loadKernel(float kernel[3][3]) {
        top->kernel_load = 1;
        top->valid_in = 1;
        
        // Load kernel column by column (3 cycles)
        for (int i = 0; i < 3; i++) {
            top->data_in0 = float_to_fp16(kernel[0][i]);
            top->data_in1 = float_to_fp16(kernel[1][i]);
            top->data_in2 = float_to_fp16(kernel[2][i]);
            clockTick();
        }
        
        // De-assert control signals
        top->kernel_load = 0;
        top->valid_in = 0;
        clockTick(); // One cycle to ensure control signals are registered
    }
    
    // Helper method to process a 3x3 image patch
    void processImage(float image[3][3]) {
        top->valid_in = 1;
        
        // Stream image data column by column (3 cycles)
        for (int i = 0; i < 3; i++) {
            top->data_in0 = float_to_fp16(image[0][i]);
            top->data_in1 = float_to_fp16(image[1][i]);
            top->data_in2 = float_to_fp16(image[2][i]);
            clockTick();
        }
        
        // De-assert valid_in after the last data is sent
        top->valid_in = 0;
    }
    
    // Helper to check if float values are close enough
    bool isClose(float a, float b, float tolerance = 0.01) {
        return std::abs(a - b) <= tolerance;
    }
};

TEST_F(ConvolutionTestbench, IdentityKernelTest)
{
    // 1. Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image
    float kernel[3][3] = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f}
    };
    float image[3][3] = {
        {10.0f, 20.0f, 30.0f},
        {40.0f, 50.0f, 60.0f},
        {70.0f, 80.0f, 90.0f}
    };
    
    // 3. Load Kernel and Process Image
    loadKernel(kernel);
    processImage(image);
    
    clockTick();         
    top->valid_out = 1; 
    clockTick();    
    top->valid_out = 0;
    // 5. Check the result
    uint16_t expected = float_to_fp16(50.0f);
    float actual = fp16_to_float(top->data_out);
    EXPECT_EQ(top->data_out, expected)
        << "Expected " << expected << " but got " << actual;

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
        {1.0f,  1.0f,  1.0f},
        {0.0f,  0.0f,  0.0f},
        {-1.0f, -1.0f, -1.0f}
    };
    float image[3][3] = {
        {10.0f, 20.0f, 30.0f},
        {40.0f, 50.0f, 60.0f},
        {70.0f, 80.0f, 90.0f}
    };
    
    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    
    // 4. Wait for pipeline
    clockTick();
    top->valid_out = 1;
    clockTick();
    top->valid_out = 0;
    
    // 5. Check result: (10+20+30) - (70+80+90) = 60 - 240 = -180
    float expected = -180.0f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_FLOAT_EQ(actual, expected)// Larger tolerance for larger numbers
        << "Expected " << expected << " but got " << actual;
}

TEST_F(ConvolutionTestbench, FloatingPointValueTest)
{
    // 1. Reset
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image
    float kernel[3][3] = {
        {0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f}
    };
    float image[3][3] = {
        {1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f}
    };

    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    
    // 4. Wait for pipeline
    clockTick();
    top->valid_out = 1;
    clockTick();
    top->valid_out = 0;
    
    // 5. Check result: 9 * 0.5 * 1.5 = 6.75
    float expected = 6.75f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_FLOAT_EQ(actual, expected)
        << "Expected " << expected << " but got " << actual;
}

TEST_F(ConvolutionTestbench, XEdgeDetectionTest)
{
    // 1. Reset
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();
    
    // 2. Define Kernel and Image
    float kernel[3][3] = {
        {1.0f, 0.0f, -1.0f},
        {2.0f, 0.0f, -2.0f},
        {1.0f, 0.0f, -1.0f}
    };
    float image[3][3] = {
        {0.0f, 1.0f, 2.0f},
        {1.0f, 2.0f, 3.0f},
        {2.0f, 3.0f, 4.0f}
    };

    // 3. Load and Process
    loadKernel(kernel);
    processImage(image);
    
    // 4. Wait for pipeline
    clockTick();
    top->valid_out = 1;
    clockTick();
    top->valid_out = 0;
    
    // 5. Check result: 9 * 0.5 * 1.5 = 6.75
    float expected = -8.0f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_FLOAT_EQ(actual, expected)
        << "Expected " << expected << " but got " << actual;
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
    tfp->open("convolution_waveform.vcd");

    // Initialize Google Test
    testing::InitGoogleTest(&argc, argv);
    int res = RUN_ALL_TESTS();

    // Cleanup
    top->final();
    tfp->close();
    delete top;
    delete tfp;

    return res;
}