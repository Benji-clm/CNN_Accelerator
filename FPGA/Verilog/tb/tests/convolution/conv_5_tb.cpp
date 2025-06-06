/*
 *  Verifies the results of the 5x5 convolution module, exits with a 0 on success.
 */

#include "base_testbench.h"
#include <cmath>  // For floating point comparisons

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

// Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern
uint16_t float_to_fp16(float value) {
    // This is a simple conversion for testbench purposes.
    // For more accurate conversion, use a library or hardware implementation.
    union { float f; uint32_t u; } v = { value };
    uint32_t f = v.u;
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f >> 13) & 0x3FF;
    if (exp <= 0) {
        exp = 0;
        frac = 0;
    } else if (exp >= 31) {
        exp = 31;
        frac = 0;
    }
    return (sign << 15) | ((exp & 0x1F) << 10) | frac;
}

// Helper: Convert FP16 bit pattern to float (approximate)
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } v = { f };
    return v.f;
}

class Convolution5Testbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->data_in0 = 0;
        top->data_in1 = 0;
        top->data_in2 = 0;
        top->data_in3 = 0;
        top->data_in4 = 0;
        top->kernel_load = 0;
        top->valid_in = 0;
        top->valid_out = 0;
        // Output: data_out
    }

    // Helper method to cycle the clock
    void clockTick(int n = 1) {
        for (int i = 0; i < n; i++) {
            top->clk = 1;
            top->eval();
            tfp->dump(ticks++);

            top->clk = 0;
            top->eval();
            tfp->dump(ticks++);
        }
    }

    // Helper method to load 5x5 kernel with FP16 conversion
    void loadKernel(float kernel[5][5]) {
        top->kernel_load = 1;
        top->valid_in = 1;

        // Load kernel row by row
        for (int i = 0; i < 5; i++) {
            top->data_in0 = float_to_fp16(kernel[i][0]);
            top->data_in1 = float_to_fp16(kernel[i][1]);
            top->data_in2 = float_to_fp16(kernel[i][2]);
            top->data_in3 = float_to_fp16(kernel[i][3]);
            top->data_in4 = float_to_fp16(kernel[i][4]);
            clockTick();
        }

        top->kernel_load = 0;
        clockTick(2); // Extra cycles to settle
    }

    // Helper method to process image data with FP16 conversion
    void processImage(float image[5][5]) {
        top->valid_in = 1;

        // Process image row by row
        for (int i = 0; i < 5; i++) {
            top->data_in0 = float_to_fp16(image[i][0]);
            top->data_in1 = float_to_fp16(image[i][1]);
            top->data_in2 = float_to_fp16(image[i][2]);
            top->data_in3 = float_to_fp16(image[i][3]);
            top->data_in4 = float_to_fp16(image[i][4]);
            clockTick();
        }

        top->valid_in = 0;
        top->valid_out = 1; // Indicate that processing is done
        clockTick(2); // Extra cycles to allow full computation
    }
    
    // Helper to check if float values are close enough (for FP comparison)
    bool isClose(float a, float b, float tolerance = 0.001) {
        return fabs(a - b) < tolerance;
    }
};

TEST_F(Convolution5Testbench, IdentityKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();

    // Identity kernel (only center element is 1)
    float kernel[5][5] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };

    // Test image data
    float image[5][5] = {
        { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f},
        { 6.0f,  7.0f,  8.0f,  9.0f, 10.0f},
        {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
        {16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
        {21.0f, 22.0f, 23.0f, 24.0f, 25.0f}
    };

    // Load the kernel
    loadKernel(kernel);

    // Process the image
    processImage(image);

    // With identity kernel, output should match the center pixel (13)
    float expected = 13.0f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_TRUE(isClose(actual, expected)) 
        << "Expected " << expected << " but got " << actual;
}


TEST_F(Convolution5Testbench, AllOnesKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();

    // All-ones kernel
    float kernel[5][5] = {
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f, 1.0f, 1.0f}
    };

    // Test image data (all 2s)
    float image[5][5] = {
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f},
        {2.0f, 2.0f, 2.0f, 2.0f, 2.0f}
    };

    // Expected result: 25 * 2 = 50
    loadKernel(kernel);
    processImage(image);
    
    float expected = 50.0f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_TRUE(isClose(actual, expected)) 
        << "Expected " << expected << " but got " << actual;
}

TEST_F(Convolution5Testbench, EdgeDetectionKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();

    // Simple horizontal edge detection kernel
    float kernel[5][5] = {
        { 1.0f,  1.0f,  1.0f,  1.0f,  1.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        { 0.0f,  0.0f,  0.0f,  0.0f,  0.0f},
        {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f}
    };

    // Test image data
    float image[5][5] = {
        { 1.0f,  2.0f,  3.0f,  4.0f,  5.0f},
        { 6.0f,  7.0f,  8.0f,  9.0f, 10.0f},
        {11.0f, 12.0f, 13.0f, 14.0f, 15.0f},
        {16.0f, 17.0f, 18.0f, 19.0f, 20.0f},
        {21.0f, 22.0f, 23.0f, 24.0f, 25.0f}
    };

    // Expected result: (1+2+3+4+5) - (21+22+23+24+25) = 15 - 115 = -100
    loadKernel(kernel);
    processImage(image);
    
    float expected = -100.0f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_TRUE(isClose(actual, expected)) 
        << "Expected " << expected << " but got " << actual;
}

// Additional test for floating point precision
TEST_F(Convolution5Testbench, FloatingPointValueTest)
{
    // Reset the module
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    clockTick();

    // Kernel with fractional values
    float kernel[5][5] = {
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
        {0.5f, 0.5f, 0.5f, 0.5f, 0.5f}
    };

    // Test image data with fractional values
    float image[5][5] = {
        {1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f, 1.5f, 1.5f},
        {1.5f, 1.5f, 1.5f, 1.5f, 1.5f}
    };

    // Expected result: 25 * 0.5 * 1.5 = 18.75
    loadKernel(kernel);
    processImage(image);
    
    float expected = 18.75f;
    float actual = fp16_to_float(top->data_out);
    EXPECT_TRUE(isClose(actual, expected)) 
        << "Expected " << expected << " but got " << actual;
}


int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("convolution5_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}