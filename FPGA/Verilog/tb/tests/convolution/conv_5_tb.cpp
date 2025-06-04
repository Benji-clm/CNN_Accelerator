/*
 *  Verifies the results of the 5x5 convolution module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

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
    void clockTick() {
        top->clk = 1;
        top->eval();
        tfp->dump(ticks++);

        top->clk = 0;
        top->eval();
        tfp->dump(ticks++);
    }

    // Helper method to load 5x5 kernel
    void loadKernel(int16_t kernel[5][5]) {
        top->kernel_load = 1;
        top->valid_in = 1;

        // Load kernel row by row
        for (int i = 0; i < 5; i++) {
            top->data_in0 = kernel[i][0];
            top->data_in1 = kernel[i][1];
            top->data_in2 = kernel[i][2];
            top->data_in3 = kernel[i][3];
            top->data_in4 = kernel[i][4];
            clockTick();
        }

        top->kernel_load = 0;
        top->valid_in = 0;
        clockTick(); // Extra cycle to settle
    }

    // Helper method to process image data
    void processImage(int16_t image[5][5]) {
        top->valid_in = 1;

        // Process image row by row
        for (int i = 0; i < 5; i++) {
            top->data_in0 = image[i][0];
            top->data_in1 = image[i][1];
            top->data_in2 = image[i][2];
            top->data_in3 = image[i][3];
            top->data_in4 = image[i][4];
            clockTick();
        }

        top->valid_in = 0;
        top->valid_out = 1; // Indicate that processing is done
        clockTick(); // Extra cycle to settle
    }
};

TEST_F(Convolution5Testbench, IdentityKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();

    // Identity kernel (only center element is 1)
    int16_t kernel[5][5] = {
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 1, 0, 0},
        {0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0}
    };

    // Test image data
    int16_t image[5][5] = {
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    // Load the kernel
    loadKernel(kernel);

    // Process the image
    processImage(image);

    // With identity kernel, output should match the center pixel (13)
    EXPECT_EQ(top->data_out, 13);
}

TEST_F(Convolution5Testbench, AllOnesKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();

    // All-ones kernel
    int16_t kernel[5][5] = {
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1}
    };

    // Test image data (all 2s)
    int16_t image[5][5] = {
        {2, 2, 2, 2, 2},
        {2, 2, 2, 2, 2},
        {2, 2, 2, 2, 2},
        {2, 2, 2, 2, 2},
        {2, 2, 2, 2, 2}
    };

    // Expected result: 25 * 2 = 50
    loadKernel(kernel);
    processImage(image);
    EXPECT_EQ(top->data_out, 50);
}

TEST_F(Convolution5Testbench, EdgeDetectionKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();

    // Simple horizontal edge detection kernel
    int16_t kernel[5][5] = {
        { 1,  1,  1,  1,  1},
        { 0,  0,  0,  0,  0},
        { 0,  0,  0,  0,  0},
        { 0,  0,  0,  0,  0},
        {-1, -1, -1, -1, -1}
    };

    // Test image data
    int16_t image[5][5] = {
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };

    // Expected result: (1+2+3+4+5) - (21+22+23+24+25) = 15 - 115 = -100
    loadKernel(kernel);
    processImage(image);
    EXPECT_EQ(top->data_out, -100);
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