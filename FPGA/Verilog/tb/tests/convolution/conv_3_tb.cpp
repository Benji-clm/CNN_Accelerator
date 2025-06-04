/*
 *  Verifies the results of the convolution module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

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
    
    // Helper method to load 3x3 kernel
    void loadKernel(int16_t kernel[3][3]) {
        top->kernel_load = 1;
        top->valid_in = 1;
        
        // Load kernel row by row
        for (int i = 0; i < 3; i++) {
            top->data_in0 = kernel[i][0];
            top->data_in1 = kernel[i][1];
            top->data_in2 = kernel[i][2];
            clockTick();
        }
        
        top->kernel_load = 0;
        top->valid_in = 0;
        clockTick(); // Extra cycle to settle
    }
    
    // Helper method to process image data
    void processImage(int16_t image[3][3]) {
        top->valid_in = 1;
        
        // Process image row by row
        for (int i = 0; i < 3; i++) {
            top->data_in0 = image[i][0];
            top->data_in1 = image[i][1];
            top->data_in2 = image[i][2];
            clockTick();
        }
        
        top->valid_in = 0;
        top->valid_out = 1; // Indicate that processing is done
        clockTick(); // Extra cycle to settle
    }
};

TEST_F(ConvolutionTestbench, IdentityKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Identity kernel (only middle element is 1)
    int16_t kernel[3][3] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    
    // Test image data
    int16_t image[3][3] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90}
    };
    
    // Load the kernel
    loadKernel(kernel);
    
    // Process the image
    processImage(image);
    
    // With identity kernel, output should match the center pixel (50)
    EXPECT_EQ(top->data_out, 50);
}


TEST_F(ConvolutionTestbench, EdgeDetectionKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Horizontal edge detection kernel
    int16_t kernel[3][3] = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
    };
    
    // Test image data
    int16_t image[3][3] = {
        {10, 20, 30},
        {40, 50, 60},
        {70, 80, 90}
    };
    
    // Expected result: (10+20+30) - (70+80+90) = 60 - 240 = -180
    
    // Load the kernel
    loadKernel(kernel);
    
    // Process the image
    processImage(image);
    
    // Check the result - negative value should be handled correctly
    EXPECT_EQ(top->data_out, -180);
}

TEST_F(ConvolutionTestbench, BlurKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Blur kernel (averaging)
    int16_t kernel[3][3] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    
    // Test image data with constant values
    int16_t image[3][3] = {
        {5, 5, 5},
        {5, 5, 5},
        {5, 5, 5}
    };
    
    // Expected result: 9 * 5 = 45
    
    // Load the kernel
    loadKernel(kernel);
    
    // Process the image
    processImage(image);
    
    // Check the result
    EXPECT_EQ(top->data_out, 45);
}


int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("convolution_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}
