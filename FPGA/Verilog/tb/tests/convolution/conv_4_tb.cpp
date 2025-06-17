/*
 * Verifies the results of the 4x4 convolution module (conv_4).
 * Exits with a 0 on success.
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
 
 // Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern
 uint16_t float_to_fp16(float value) {
     half h(value);
     return h.bits();
 }
 
 // Helper: Convert FP16 bit pattern to float
 float fp16_to_float(uint16_t value) {
     half h;
     h.setBits(value);
     return (float)h;
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
         top->data_in[3] = 0;
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
     
     // Helper method to load a 4x4 kernel row by row
     void loadKernel(float kernel[4][4]) {
         top->kernel_load = 1;
         top->valid_in = 1;
         
         // Load kernel row by row (4 cycles)
         for (int i = 0; i < 4; i++) {
             top->data_in[0] = float_to_fp16(kernel[i][0]);
             top->data_in[1] = float_to_fp16(kernel[i][1]);
             top->data_in[2] = float_to_fp16(kernel[i][2]);
             top->data_in[3] = float_to_fp16(kernel[i][3]);
             clockTick();
         }
         
         // De-assert control signals
         top->kernel_load = 0;
         top->valid_in = 0;
         clockTick(); // One cycle to ensure control signals are registered
     }
     
     // Helper method to process a 4x4 image patch row by row
     void processImage(float image[4][4]) {
         top->valid_in = 1;
         
         // Stream image data row by row (4 cycles)
         for (int i = 0; i < 4; i++) {
             top->data_in[0] = float_to_fp16(image[i][0]);
             top->data_in[1] = float_to_fp16(image[i][1]);
             top->data_in[2] = float_to_fp16(image[i][2]);
             top->data_in[3] = float_to_fp16(image[i][3]);
             clockTick();
         }
         
         // De-assert valid_in after the last data is sent
         top->valid_in = 0;
     }
 };
 
 TEST_F(ConvolutionTestbench, IdentityKernelTest)
 {
     // 1. Reset the module
     top->rst = 1;
     clockTick(2);
     top->rst = 0;
     clockTick();
     
     // 2. Define 4x4 Kernel and Image
     float kernel[4][4] = {
         {0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 1.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f},
         {0.0f, 0.0f, 0.0f, 0.0f}
     };
     float image[4][4] = {
         {1.0f, 2.0f, 3.0f, 4.0f},
         {5.0f, 6.0f, 7.0f, 8.0f},
         {9.0f, 10.0f, 11.0f, 12.0f},
         {13.0f, 14.0f, 15.0f, 16.0f}
     };
     
     // 3. Load Kernel and Process Image
     loadKernel(kernel);
     processImage(image);
     
     // 4. Latch the output
     clockTick();
     top->valid_out = 1; 
     clockTick();
     top->valid_out = 0;
     
     // 5. Check the result (should be image[1][1])
     float expected = 6.0f;
     float actual = fp16_to_float(top->data_out);
     EXPECT_FLOAT_EQ(actual, expected)
         << "Expected " << expected << " but got " << actual;
 }
 
 TEST_F(ConvolutionTestbench, VerticalEdgeDetectionTest)
 {
     // 1. Reset
     top->rst = 1;
     clockTick(2);
     top->rst = 0;
     clockTick();
     
     // 2. Define 4x4 Kernel and Image
     float kernel[4][4] = {
         { 1.0f,  1.0f,  1.0f,  1.0f},
         { 1.0f,  1.0f,  1.0f,  1.0f},
         {-1.0f, -1.0f, -1.0f, -1.0f},
         {-1.0f, -1.0f, -1.0f, -1.0f}
     };
     float image[4][4] = {
         {10.0f, 20.0f, 30.0f, 40.0f}, // Sum = 100
         {50.0f, 60.0f, 70.0f, 80.0f}, // Sum = 260
         {1.0f,  2.0f,  3.0f,  4.0f},  // Sum = 10
         {5.0f,  6.0f,  7.0f,  8.0f}   // Sum = 26
     };
     
     // 3. Load and Process
     loadKernel(kernel);
     processImage(image);
     
     // 4. Latch the output
     clockTick();
     top->valid_out = 1;
     clockTick();
     top->valid_out = 0;
     
     // 5. Check result: (100 + 260) - (10 + 26) = 360 - 36 = 324
     float expected = 324.0f;
     float actual = fp16_to_float(top->data_out);
     EXPECT_FLOAT_EQ(actual, expected)
         << "Expected " << expected << " but got " << actual;
 }
 
 TEST_F(ConvolutionTestbench, FloatingPointValueTest)
 {
     // 1. Reset
     top->rst = 1;
     clockTick(2);
     top->rst = 0;
     clockTick();
     
     // 2. Define 4x4 Kernel and Image
     float kernel[4][4] = {
         {0.5f, 0.5f, 0.5f, 0.5f},
         {0.5f, 0.5f, 0.5f, 0.5f},
         {0.5f, 0.5f, 0.5f, 0.5f},
         {0.5f, 0.5f, 0.5f, 0.5f}
     };
     float image[4][4] = {
         {1.5f, 1.5f, 1.5f, 1.5f},
         {1.5f, 1.5f, 1.5f, 1.5f},
         {1.5f, 1.5f, 1.5f, 1.5f},
         {1.5f, 1.5f, 1.5f, 1.5f}
     };
 
     // 3. Load and Process
     loadKernel(kernel);
     processImage(image);
     
     // 4. Latch the output
     clockTick();
     top->valid_out = 1;
     clockTick();
     top->valid_out = 0;
     
     // 5. Check result: 16 * (0.5 * 1.5) = 12.0
     float expected = 12.0f;
     float actual = fp16_to_float(top->data_out);
     EXPECT_FLOAT_EQ(actual, expected)
         << "Expected " << expected << " but got " << actual;
 }
 
 TEST_F(ConvolutionTestbench, XEdgeDetectionSobelTest)
 {
     // 1. Reset
     top->rst = 1;
     clockTick(2);
     top->rst = 0;
     clockTick();
     
     // 2. Define 4x4 Kernel and Image (Sobel X-operator)
     float kernel[4][4] = {
         {1.0f, 0.0f, -1.0f, 0.0f},
         {2.0f, 0.0f, -2.0f, 0.0f},
         {2.0f, 0.0f, -2.0f, 0.0f},
         {1.0f, 0.0f, -1.0f, 0.0f}
     };
     float image[4][4] = {
         {1.0f, 2.0f, 3.0f, 4.0f},
         {5.0f, 6.0f, 7.0f, 8.0f},
         {9.0f, 10.0f, 11.0f, 12.0f},
         {13.0f, 14.0f, 15.0f, 16.0f}
     };
 
     // 3. Load and Process
     loadKernel(kernel);
     processImage(image);
     
     // 4. Latch the output
     clockTick();
     top->valid_out = 1;
     clockTick();
     top->valid_out = 0;
     
     // 5. Check result
     // Col0: 1*1 + 5*2 + 9*2 + 13*1 = 42
     // Col2: 3*(-1) + 7*(-2) + 11*(-2) + 15*(-1) = -54
     // Total = 42 - 54 = -12
     float expected = -12.0f;
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
     tfp->open("convolution_4_waveform.vcd");
 
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