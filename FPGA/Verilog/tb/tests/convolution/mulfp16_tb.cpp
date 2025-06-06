// filepath: /root/Documents/iac/lab0-devtools/autumn/workspace/CNN_Accelerator/FPGA/Verilog/tb/tests/test_fp16_max_comparator_tb.cpp
/*
 *  Verifies the results of the fp16_max_comparator module, exits with a 0 on success.
 */

 #include "base_testbench.h"
 
 Vdut *top;
 VerilatedVcdC *tfp;
 unsigned int ticks = 0;
 
 class FP16MultiplierTestbench : public BaseTestbench {
    protected:
        void initializeInputs() override {
            top->a_in = 0;
            top->b_in = 0;
        }
    
        // Set inputs and evaluate the combinational logic
        void setInputs(uint16_t a_val, uint16_t b_val) {
            top->a_in = a_val;
            top->b_in = b_val;
            top->eval();
            if (tfp) tfp->dump(ticks++);
        }
    };
    
    TEST_F(FP16MultiplierTestbench, PositiveNumbers) {
        // Test 1: a = 1.0 (0x3C00), b = 2.0 (0x4000), expect c_out = 2.0 (0x4000)
        setInputs(0x3C00, 0x4000);
        EXPECT_EQ(top->c_out, 0x4000);
    
        // Test 2: a = 2.0 (0x4000), b = 3.0 (0x4200), expect c_out = 6.0 (0x4600)
        setInputs(0x4000, 0x4200);
        EXPECT_EQ(top->c_out, 0x4600);
    
        // Test 3: a = 1.5 (0x3E00), b = 1.5 (0x3E00), expect c_out = 2.25 (0x4080)
        setInputs(0x3E00, 0x3E00);
        EXPECT_EQ(top->c_out, 0x4080);
    }
    
    TEST_F(FP16MultiplierTestbench, NegativeNumbers) {
        // Test 1: a = -1.0 (0xBC00), b = 2.0 (0x4000), expect c_out = -2.0 (0xC000)
        setInputs(0xBC00, 0x4000);
        EXPECT_EQ(top->c_out, 0xC000);
    
        // Test 2: a = -1.0 (0xBC00), b = -2.0 (0xC000), expect c_out = 2.0 (0x4000)
        setInputs(0xBC00, 0xC000);
        EXPECT_EQ(top->c_out, 0x4000);
    }
    
    TEST_F(FP16MultiplierTestbench, Zero) {
        // Test 1: a = 0.0 (0x0000), b = 1.0 (0x3C00), expect c_out = 0.0 (0x0000)
        setInputs(0x0000, 0x3C00);
        EXPECT_EQ(top->c_out, 0x0000);
    
        // Test 2: a = -0.0 (0x8000), b = 1.0 (0x3C00), expect c_out = -0.0 (0x8000)
        setInputs(0x8000, 0x3C00);
        EXPECT_EQ(top->c_out, 0x8000); // Assuming module preserves -0.0
    }
    
    TEST_F(FP16MultiplierTestbench, SubnormalNumbers) {
        // Test 1: a = 2^(-24) (0x0001), b = 1.0 (0x3C00), expect c_out = 2^(-24) (0x0001)
        setInputs(0x0001, 0x3C00);
        EXPECT_EQ(top->c_out, 0x0001);
    
        // Test 2: a = 2^(-10) (0x1400), b = 2^(-10) (0x1400), expect c_out = 2^(-20) (0x0010)
        setInputs(0x1400, 0x1400);
        EXPECT_EQ(top->c_out, 0x0010);
    }
    
    TEST_F(FP16MultiplierTestbench, Overflow) {
        // Test 1: a = 65504 (0x7BFF), b = 65504 (0x7BFF), expect c_out = +inf (0x7C00)
        setInputs(0x7BFF, 0x7BFF);
        EXPECT_EQ(top->c_out, 0x7C00);
    
        // Test 2: a = -65504 (0xFBFF), b = -65504 (0xFBFF), expect c_out = +inf (0x7C00)
        setInputs(0xFBFF, 0xFBFF);
        EXPECT_EQ(top->c_out, 0x7C00);
    }
    
    TEST_F(FP16MultiplierTestbench, Underflow) {
        // Test 1: a = 2^(-14) (0x0400), b = 2^(-14) (0x0400), expect c_out = 0.0 (0x0000)
        setInputs(0x0400, 0x0400);
        EXPECT_EQ(top->c_out, 0x0000); // Result 2^(-28) underflows to zero
    }
    
    int main(int argc, char **argv) {
        top = new Vdut;
        tfp = new VerilatedVcdC;
    
        Verilated::traceEverOn(true);
        top->trace(tfp, 99);
        tfp->open("fp16_multiplier_waveform.vcd");
    
        testing::InitGoogleTest(&argc, argv);
        auto res = RUN_ALL_TESTS();
    
        top->final();
        tfp->close();
    
        delete top;
        delete tfp;
    
        return res;
    }