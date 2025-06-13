/*
*  Verifies the results of the convolution module, exits with a 0 on success.
*/

#include "../testbench.h"
#include <cmath>  // For floating point comparisons
#include <Imath/half.h>

using Imath::half;

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

uint16_t float_to_fp16(float value) {
    half h(value); // Implicit conversion from float to half
    return h.bits(); // Safe bit pattern extraction
}

float fp16_to_float(uint16_t value) {
    half h;
    h.setBits(value); // Safe bit pattern assignment
    return (float)h;
}

class PoolingTestbench : public Testbench
{
protected:
    void initializeInputs() override
    {   
        top->clk = 0;
        top->rst = 1;
        top->input_column = 0;
        top->valid_in = 0;
    }
};

TEST_F(PoolingTestbench, BasicMaxPooling)
{
    top->rst = 0;
    // First input cycle: [1.0, 2.0]
    uint16_t a00 = float_to_fp16(1.0);
    uint16_t a01 = float_to_fp16(2.0);
    top->input_column = (static_cast<uint32_t>(a01) << 16) | static_cast<uint32_t>(a00);
    top->valid_in = 1;
    runSimulation(1);

    // Invalid cycle: [3.0, 4.0]
    top->valid_in = 0;
    uint16_t b0 = float_to_fp16(3.0);
    uint16_t b1 = float_to_fp16(4.0);
    top->input_column = (static_cast<uint32_t>(b0) << 16) | static_cast<uint32_t>(b1);
    top->eval();  // Add this
    EXPECT_EQ(top->valid_out, 0);
    runSimulation(1);

    // Second input cycle: [0.0, 0.0]
    top->valid_in = 1;
    uint16_t a10 = float_to_fp16(0.0);
    uint16_t a11 = float_to_fp16(0.0);
    top->input_column = (static_cast<uint32_t>(a10) << 16) | static_cast<uint32_t>(a11);
    top->eval();  // Add this
    EXPECT_EQ(top->valid_out, 1) << "valid_out should be high after two valid inputs";
    float expected = 2.0;
    float actual = fp16_to_float(top->output_column);
    EXPECT_FLOAT_EQ(actual, expected) << "Expected max = 2.0, got " << actual;
}

TEST_F(PoolingTestbench, SecondMax)
{
    top->rst = 0;
    // First input cycle: [1.0, 2.0]
    uint16_t a00 = float_to_fp16(1.0);
    uint16_t a01 = float_to_fp16(2.0);
    top->input_column = (static_cast<uint32_t>(a01) << 16) | static_cast<uint32_t>(a00);
    top->valid_in = 1;
    runSimulation(1);

    // Invalid cycle: [3.0, 4.0]
    top->valid_in = 0;
    uint16_t b0 = float_to_fp16(3.0);
    uint16_t b1 = float_to_fp16(4.0);
    top->input_column = (static_cast<uint32_t>(b0) << 16) | static_cast<uint32_t>(b1);
    top->eval();  // Add this
    EXPECT_EQ(top->valid_out, 0);
    runSimulation(1);

    // Second input cycle: [0.0, 0.0]
    top->valid_in = 1;
    uint16_t a10 = float_to_fp16(10.0);
    uint16_t a11 = float_to_fp16(0.0);
    top->input_column = (static_cast<uint32_t>(a10) << 16) | static_cast<uint32_t>(a11);
    top->eval();  // Add this
    EXPECT_EQ(top->valid_out, 1) << "valid_out should be high after two valid inputs";
    float expected = 10.0;
    float actual = fp16_to_float(top->output_column);
    EXPECT_FLOAT_EQ(actual, expected) << "Expected max = 10.0, got " << actual;
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