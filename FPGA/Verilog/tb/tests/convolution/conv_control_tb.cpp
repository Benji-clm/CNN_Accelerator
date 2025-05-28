/*
 *  Verifies the results of the parallel_convolution module, exits with a 0 on success.
 */

#include "base_testbench.h"
#include <vector>

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class ParallelConvolutionTestbench : public BaseTestbench
{
protected:
    static const int IMAGE_SIZE = 6;
    static const int KERNEL_SIZE = 3;
    static const int OUTPUT_SIZE = 4; // (IMAGE_SIZE - KERNEL_SIZE + 2*PADDING)/STRIDE + 1
    static const int NUM_PARALLEL = 4;

    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->start = 0;
        // Initialize the 2D arrays
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                top->data_in[i][j] = 0;
            }
        }
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                top->kernel_in[i][j] = 0;
            }
        }
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
    
    // Helper to load test data
    void loadTestData(int16_t kernel[KERNEL_SIZE][KERNEL_SIZE], int16_t image[IMAGE_SIZE][IMAGE_SIZE]) {
        // Load kernel data
        for (int i = 0; i < KERNEL_SIZE; i++) {
            for (int j = 0; j < KERNEL_SIZE; j++) {
                top->kernel_in[i][j] = kernel[i][j];
            }
        }
        
        // Load image data
        for (int i = 0; i < IMAGE_SIZE; i++) {
            for (int j = 0; j < IMAGE_SIZE; j++) {
                top->data_in[i][j] = image[i][j];
            }
        }
    }
    
    // Helper to verify results
    void verifyResults(int32_t expected[OUTPUT_SIZE][OUTPUT_SIZE]) {
        for (int i = 0; i < OUTPUT_SIZE; i++) {
            for (int j = 0; j < OUTPUT_SIZE; j++) {
                EXPECT_EQ(top->result_out[i][j], expected[i][j]) 
                    << "Mismatch at position [" << i << "][" << j << "]";
            }
        }
    }
};

TEST_F(ParallelConvolutionTestbench, SimpleIdentityKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Identity kernel (only middle element is 1)
    int16_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {0, 0, 0},
        {0, 1, 0},
        {0, 0, 0}
    };
    
    // Test image data (6x6 for simplicity)
    int16_t image[IMAGE_SIZE][IMAGE_SIZE] = {
        {10, 20, 30, 40, 50, 60},
        {15, 25, 35, 45, 55, 65},
        {20, 30, 40, 50, 60, 70},
        {25, 35, 45, 55, 65, 75},
        {30, 40, 50, 60, 70, 80},
        {35, 45, 55, 65, 75, 85}
    };
    
    // Expected result for 4x4 output (identity kernel selects center pixel)
    int32_t expected[OUTPUT_SIZE][OUTPUT_SIZE] = {
        {25, 35, 45, 55},
        {30, 40, 50, 60},
        {35, 45, 55, 65},
        {40, 50, 60, 70}
    };
    
    // Load test data
    loadTestData(kernel, image);
    
    // Start convolution
    top->start = 1;
    clockTick();
    top->start = 0;
    
    // Wait for convolution to complete (should check for done signal)
    int max_cycles = 100;
    while (!top->done && max_cycles > 0) {
        clockTick();
        max_cycles--;
    }
    
    // Verify that convolution completed
    ASSERT_TRUE(top->done) << "Convolution did not complete within expected time";
    
    // Verify results
    verifyResults(expected);
}

TEST_F(ParallelConvolutionTestbench, EdgeDetectionKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Horizontal edge detection kernel
    int16_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1, 1, 1},
        {0, 0, 0},
        {-1, -1, -1}
    };
    
    // Test image data (6x6)
    int16_t image[IMAGE_SIZE][IMAGE_SIZE] = {
        {10, 10, 10, 10, 10, 10},
        {10, 10, 10, 10, 10, 10},
        {10, 10, 10, 10, 10, 10},
        {50, 50, 50, 50, 50, 50},
        {50, 50, 50, 50, 50, 50},
        {50, 50, 50, 50, 50, 50}
    };
    
    // Expected output for edge detection (strong horizontal edge in the middle)
    int32_t expected[OUTPUT_SIZE][OUTPUT_SIZE] = {
        {0, 0, 0, 0},
        {120, 120, 120, 120}, // Edge detected (10*3 - 50*3) * 3 positions
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };
    
    // Load test data
    loadTestData(kernel, image);
    
    // Start convolution
    top->start = 1;
    clockTick();
    top->start = 0;
    
    // Wait for convolution to complete
    int max_cycles = 100;
    while (!top->done && max_cycles > 0) {
        clockTick();
        max_cycles--;
    }
    
    // Verify that convolution completed
    ASSERT_TRUE(top->done) << "Convolution did not complete within expected time";
    
    // Verify results
    verifyResults(expected);
}

TEST_F(ParallelConvolutionTestbench, BlurKernelTest)
{
    // Reset the module
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Blur kernel (averaging)
    int16_t kernel[KERNEL_SIZE][KERNEL_SIZE] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}
    };
    
    // Test image with constant values
    int16_t image[IMAGE_SIZE][IMAGE_SIZE];
    for (int i = 0; i < IMAGE_SIZE; i++) {
        for (int j = 0; j < IMAGE_SIZE; j++) {
            image[i][j] = 5;
        }
    }
    
    // Expected output for blur kernel (each output is 9*5 = 45)
    int32_t expected[OUTPUT_SIZE][OUTPUT_SIZE];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            expected[i][j] = 45;
        }
    }
    
    // Load test data
    loadTestData(kernel, image);
    
    // Start convolution
    top->start = 1;
    clockTick();
    top->start = 0;
    
    // Wait for convolution to complete
    int max_cycles = 100;
    while (!top->done && max_cycles > 0) {
        clockTick();
        max_cycles--;
    }
    
    // Verify that convolution completed
    ASSERT_TRUE(top->done) << "Convolution did not complete within expected time";
    
    // Verify results
    verifyResults(expected);
}

int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("parallel_convolution_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}