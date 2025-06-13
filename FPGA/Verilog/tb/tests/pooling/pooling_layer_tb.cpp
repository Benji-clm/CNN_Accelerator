/*
 * Verifies the results of the convolution module, exits with a 0 on success.
 */

 #include "../testbench.h"
 #include <cmath>        // For floating point comparisons
 #include <Imath/half.h>
 #include <array>        // For std::array
 
 using Imath::half;
 
 Vdut *top;
 VerilatedVcdC *tfp;
 unsigned int ticks = 0;
 
 // Helper function to convert a float to a 16-bit floating point representation
 uint16_t float_to_fp16(float value) {
     half h(value); // Implicit conversion from float to half
     return h.bits(); // Safe bit pattern extraction
 }
 
 // Helper function to convert a 16-bit floating point back to a float
 float fp16_to_float(uint16_t value) {
     half h;
     h.setBits(value); // Safe bit pattern assignment
     return (float)h;
 }
 
 class PoolingTestbench : public Testbench
 {
 protected:
     // Initializes all DUT inputs to a known state
     void initializeInputs() override {
         top->clk = 0;
         top->rst = 1;
         for (int i = 0; i < 24; i++) {
             top->input_column[i] = 0;
         }
         top->valid_in = 0;
     }
 
     void drive_wide_column(const std::array<uint16_t, 24>& column_data) {
        for (int i = 0; i < 12; ++i) {
            uint32_t low_word = column_data[i * 2];
            uint32_t high_word = column_data[i * 2 + 1];
            top->input_column[i] = (high_word << 16) | low_word;
        }
    }
 };
 
 TEST_F(PoolingTestbench, BasicMaxPooling) {
     top->rst = 0;
 
     // First input cycle: Set 24 elements [1.0, 2.0, 1.0, 2.0, ...]
     std::array<uint16_t, 24> input_first;
     for (int i = 0; i < 24; i += 2) {
         input_first[i] = float_to_fp16(1.0);
         input_first[i + 1] = float_to_fp16(2.0);
     }
     drive_wide_column(input_first);
     top->valid_in = 1;
     runSimulation(1);
 
     // Invalid cycle: Set 24 elements [3.0, 4.0, 3.0, 4.0, ...]
     top->valid_in = 0;
     std::array<uint16_t, 24> input_invalid;
     for (int i = 0; i < 24; i += 2) {
         input_invalid[i] = float_to_fp16(3.0);
         input_invalid[i + 1] = float_to_fp16(4.0);
     }
     drive_wide_column(input_invalid);
     top->eval();
     EXPECT_EQ(top->valid_out, 0) << "valid_out should be low during invalid cycle";
     runSimulation(1);
 
     // Second input cycle: Set 24 elements to all 0.0
     top->valid_in = 1;
     std::array<uint16_t, 24> input_second;
     input_second.fill(float_to_fp16(0.0));
     drive_wide_column(input_second);
     runSimulation(1); // Run for one cycle to allow output to be valid
     
     EXPECT_EQ(top->valid_out, 1) << "valid_out should be high after two valid inputs";
 
     // Check all 12 output elements
     for (int i = 0; i < 12; i++) {
         float expected = 2.0; // Max of [1.0, 2.0, 0.0, 0.0] for each pooling unit
         float actual = fp16_to_float(top->output_column[i]);
         EXPECT_FLOAT_EQ(actual, expected) << "For output_column[" << i << "], expected max = 2.0, got " << actual;
     }
 }
 
 TEST_F(PoolingTestbench, SecondMax) {
     top->rst = 0;
     runSimulation(2);
 
     // First input cycle: Set 24 elements [1.0, 2.0, 1.0, 2.0, ...]
     // FIXED: The input_first array was previously uninitialized.
     std::array<uint16_t, 24> input_first;
     for (int i = 0; i < 24; i += 2) {
         input_first[i] = float_to_fp16(1.0);
         input_first[i + 1] = float_to_fp16(2.0);
     }
     drive_wide_column(input_first);
     top->valid_in = 1;
     runSimulation(1);
 
     // Invalid cycle: Set 24 elements [3.0, 4.0, 3.0, 4.0, ...]
     top->valid_in = 0;
     std::array<uint16_t, 24> input_invalid;
     for (int i = 0; i < 24; i += 2) {
         input_invalid[i] = float_to_fp16(3.0);
         input_invalid[i + 1] = float_to_fp16(4.0);
     }
     drive_wide_column(input_invalid);
     top->eval();
     EXPECT_EQ(top->valid_out, 0) << "valid_out should be low during invalid cycle";
     runSimulation(1);
 
     // Second input cycle: Set 24 elements [10.0, 1.0, 10.0, 1.0, ...]
     top->valid_in = 1;
     std::array<uint16_t, 24> input_second;
     for (int i = 0; i < 24; i += 2) {
         input_second[i] = float_to_fp16(10.0);
         input_second[i + 1] = float_to_fp16(1.0);
     }
     drive_wide_column(input_second);
     runSimulation(1);
     EXPECT_EQ(top->valid_out, 1) << "valid_out should be high after two valid inputs";
 
     // Check all 12 output elements
     for (int i = 0; i < 12; i++) {
         // The pooling unit sees inputs (1.0, 2.0) from the first cycle and (10.0, 1.0) from this cycle.
         float expected = 10.0; // Max of {1.0, 2.0, 10.0, 1.0}
         float actual = fp16_to_float(top->output_column[i]);
         EXPECT_FLOAT_EQ(actual, expected) << "For output_column[" << i << "], expected max = 10.0, got " << actual;
     }
     top->valid_in = 0;
     runSimulation(2);
 }
 
 int main(int argc, char **argv)
 {
     top = new Vdut;
     tfp = new VerilatedVcdC;
 
     Verilated::traceEverOn(true);
     top->trace(tfp, 99);
     tfp->open("pooling_layer.vcd");
 
     testing::InitGoogleTest(&argc, argv);
     auto res = RUN_ALL_TESTS();
     top->final();
     tfp->close();
     
     delete top;
     delete tfp;
 
     return res;
 }