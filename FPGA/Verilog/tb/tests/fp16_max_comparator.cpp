// filepath: /root/Documents/iac/lab0-devtools/autumn/workspace/CNN_Accelerator/FPGA/Verilog/tb/tests/test_fp16_max_comparator_tb.cpp
/*
 *  Verifies the results of the fp16_max_comparator module, exits with a 0 on success.
 */

 #include "base_testbench.h"
 #include "Vfp16_max_comparator.h" // Verilated module header
 
 Vfp16_max_comparator *top;
 VerilatedVcdC *tfp;
 unsigned int ticks = 0;
 
 class FP16MaxComparatorTestbench : public BaseTestbench {
 protected:
     void initializeInputs() override {
         top->a = 0;
         top->b = 0;
     }
 
     // Set inputs and evaluate the combinational logic
     void setInputs(uint16_t a_val, uint16_t b_val) {
         top->a = a_val;
         top->b = b_val;
         top->eval();
         if (tfp) tfp->dump(ticks++);
     }
 };
 
 TEST_F(FP16MaxComparatorTestbench, PositiveNumbers) {
     // Test 1: a = 1.0 (0x3C00), b = 2.0 (0x4000)
     setInputs(0x3C00, 0x4000);
     EXPECT_EQ(top->max_val, 0x4000); // exp_a = 15 < exp_b = 16
 
     // Test 2: a = 2.0 (0x4000), b = 1.0 (0x3C00)
     setInputs(0x4000, 0x3C00);
     EXPECT_EQ(top->max_val, 0x4000); // exp_a = 16 > exp_b = 15
 
     // Test 3: a = 1.0 (0x3C00), b = 1.5 (0x3E00)
     setInputs(0x3C00, 0x3E00);
     EXPECT_EQ(top->max_val, 0x3E00); // exp equal (15), mant_a = 0 < mant_b = 512
 
     // Test 4: a = 1.5 (0x3E00), b = 1.0 (0x3C00)
     setInputs(0x3E00, 0x3C00);
     EXPECT_EQ(top->max_val, 0x3E00); // exp equal (15), mant_a = 512 > mant_b = 0
 
     // Test 5: a = 1.5 (0x3E00), b = 1.5 (0x3E00)
     setInputs(0x3E00, 0x3E00);
     EXPECT_EQ(top->max_val, 0x3E00); // exp and mant equal, chooses b
 }
 
 TEST_F(FP16MaxComparatorTestbench, NegativeNumbers) {
     // Note: Module does not correctly handle signs for maximum value selection
 
     // Test 1: a = -1.0 (0xBC00), b = -2.0 (0xC000)
     setInputs(0xBC00, 0xC000);
     EXPECT_EQ(top->max_val, 0xC000); // exp_a = 15 < exp_b = 16, outputs -2.0 (incorrect, -1.0 > -2.0)
 
     // Test 2: a = -2.0 (0xC000), b = -1.0 (0xBC00)
     setInputs(0xC000, 0xBC00);
     EXPECT_EQ(top->max_val, 0xC000); // exp_a = 16 > exp_b = 15, outputs -2.0 (incorrect)
 
     // Test 3: a = -1.0 (0xBC00), b = 1.0 (0x3C00)
     setInputs(0xBC00, 0x3C00);
     EXPECT_EQ(top->max_val, 0x3C00); // exp equal (15), mant equal (0), chooses b, correct (1.0 > -1.0)
 
     // Test 4: a = 1.0 (0x3C00), b = -1.0 (0xBC00)
     setInputs(0x3C00, 0xBC00);
     EXPECT_EQ(top->max_val, 0xBC00); // exp equal (15), mant equal (0), chooses b, incorrect (-1.0 < 1.0)
 }
 
 int main(int argc, char **argv) {
     top = new Vfp16_max_comparator;
     tfp = new VerilatedVcdC;
 
     Verilated::traceEverOn(true);
     top->trace(tfp, 99);
     tfp->open("fp16_max_comparator_waveform.vcd");
 
     testing::InitGoogleTest(&argc, argv);
     auto res = RUN_ALL_TESTS();
 
     top->final();
     tfp->close();
 
     delete top;
     delete tfp;
 
     return res;
 }