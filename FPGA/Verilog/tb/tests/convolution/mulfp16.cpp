/*
 *  Verifies the results of the ProcessingElement module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

// Helper functions for FP16 operations (8-bit exponent version used in PE)
uint16_t pe_fp16_pack(bool sign, uint8_t exp, uint8_t frac) {
    return (sign << 15) | (exp << 7) | frac;
}

// Define common constants
const uint16_t PE_FP16_ZERO = pe_fp16_pack(0, 0, 0);
const uint16_t PE_FP16_ONE = pe_fp16_pack(0, 127, 0);
const uint16_t PE_FP16_TWO = pe_fp16_pack(0, 128, 0);
const uint16_t PE_FP16_HALF = pe_fp16_pack(0, 126, 0);
const uint16_t PE_FP16_MINUS_ONE = pe_fp16_pack(1, 127, 0);

class ProcessingElementTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->load_b = 0;
        top->mode_fp16 = 0;
        top->signed_mode = 0;
        top->a_in = 0;
        top->b_in = 0;
        top->c_in = 0;
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
    
    // Helper method to apply inputs and check outputs for integer mode
    void applyAndCheckInteger(uint8_t a, uint8_t b, uint8_t c_in, uint16_t expected_a_out, uint16_t expected_c_out, bool is_signed = false) {
        // Set inputs
        top->a_in = a;
        top->b_in = b;
        top->c_in = c_in;
        top->mode_fp16 = 0;
        top->signed_mode = is_signed;
        
        // Load weight (b_in)
        top->load_b = 1;
        clockTick();
        top->load_b = 0;
        
        // Process the computation
        clockTick();
        
        // Check outputs
        EXPECT_EQ(top->a_out, expected_a_out) << "a_out mismatch in integer mode";
        EXPECT_EQ(top->c_out, expected_c_out) << "c_out mismatch in integer mode";
    }
    
    // Helper method to apply inputs and check outputs for FP16 mode
    void applyAndCheckFP16(uint16_t a, uint16_t b, uint16_t c_in, uint16_t expected_a_out, uint16_t expected_c_out) {
        // Set inputs
        top->a_in = a;
        top->b_in = b;
        top->c_in = c_in;
        top->mode_fp16 = 1;
        
        // Load weight (b_in)
        top->load_b = 1;
        clockTick();
        top->load_b = 0;
        
        // Process the computation
        clockTick();
        
        // Check outputs
        EXPECT_EQ(top->a_out, expected_a_out) << "a_out mismatch in FP16 mode";
        EXPECT_EQ(top->c_out, expected_c_out) << "c_out mismatch in FP16 mode";
    }
};

TEST_F(ProcessingElementTestbench, BasicIntegerMultiply)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 5 * 4 + 0 = 20 (unsigned)
    applyAndCheckInteger(5, 4, 0, 5, 20, false);
}

TEST_F(ProcessingElementTestbench, IntegerMultiplyWithAccumulation)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 10 * 7 + 15 = 85 (unsigned)
    applyAndCheckInteger(10, 7, 15, 10, 85, false);
}

TEST_F(ProcessingElementTestbench, SignedIntegerMultiply)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: -5 * 4 + 0 = -20 (signed)
    // -5 in 8-bit signed = 251 (0xFB)
    applyAndCheckInteger(0xFB, 4, 0, 0xFB, 0xFFEC, true); // 0xFFEC = -20 in 16-bit signed
}a

TEST_F(ProcessingElementTestbench, SignedIntegerMultiplyWithAccumulation)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: -3 * -6 + 10 = 28 (signed)
    // -3 in 8-bit signed = 253 (0xFD)
    // -6 in 8-bit signed = 250 (0xFA)
    applyAndCheckInteger(0xFD, 0xFA, 10, 0xFD, 28, true);
}

TEST_F(ProcessingElementTestbench, FP16MultiplyOneByOne)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 1.0 * 1.0 + 0.0 = 1.0
    applyAndCheckFP16(PE_FP16_ONE, PE_FP16_ONE, PE_FP16_ZERO, PE_FP16_ONE, PE_FP16_ONE);
}

TEST_F(ProcessingElementTestbench, FP16MultiplyWithAccumulation)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 2.0 * 0.5 + 1.0 = 2.0
    applyAndCheckFP16(PE_FP16_TWO, PE_FP16_HALF, PE_FP16_ONE, PE_FP16_TWO, PE_FP16_TWO);
}

TEST_F(ProcessingElementTestbench, FP16MultiplyNegative)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: -1.0 * 2.0 + 0.0 = -2.0
    uint16_t expected_result = pe_fp16_pack(1, 128, 0); // -2.0
    applyAndCheckFP16(PE_FP16_MINUS_ONE, PE_FP16_TWO, PE_FP16_ZERO, PE_FP16_MINUS_ONE, expected_result);
}

TEST_F(ProcessingElementTestbench, FP16AddOppositeSignsSmaller)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 1.0 * 1.0 + (-0.5) = 0.5
    uint16_t minus_half = pe_fp16_pack(1, 126, 0); // -0.5
    uint16_t half = pe_fp16_pack(0, 126, 0); // 0.5
    applyAndCheckFP16(PE_FP16_ONE, PE_FP16_ONE, minus_half, PE_FP16_ONE, half);
}

TEST_F(ProcessingElementTestbench, FP16AddOppositeSignsLarger)
{
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Test case: 0.5 * 1.0 + (-1.0) = -0.5
    uint16_t minus_half = pe_fp16_pack(1, 126, 0); // -0.5
    applyAndCheckFP16(PE_FP16_HALF, PE_FP16_ONE, PE_FP16_MINUS_ONE, PE_FP16_HALF, minus_half);
}

TEST_F(ProcessingElementTestbench, StateRetention)
{
    // This test verifies that the PE retains its weight value (b) between computations
    
    // Reset the PE
    top->rst = 1;
    clockTick();
    top->rst = 0;
    clockTick();
    
    // Load weight (b_in = 4)
    top->b_in = 4;
    top->mode_fp16 = 0;
    top->signed_mode = 0;
    top->load_b = 1;
    clockTick();
    top->load_b = 0;
    
    // First computation: 5 * 4 + 0 = 20
    top->a_in = 5;
    top->c_in = 0;
    clockTick();
    EXPECT_EQ(top->a_out, 5);
    EXPECT_EQ(top->c_out, 20);
    
    // Second computation with same weight but different a_in: 10 * 4 + 0 = 40
    top->a_in = 10;
    top->c_in = 0;
    clockTick();
    EXPECT_EQ(top->a_out, 10);
    EXPECT_EQ(top->c_out, 40);
}

int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("processing_element_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}