/*
*  Verifies the results of the convolution module, exits with a 0 on success.
*/

#include "../base_testbench.h"
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

class PoolingTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst = 1;
        top->a = 0;
        top->b = 0;
        top->store = 0;
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
};

TEST_F(PoolingTestbench, A_max)
{
    // Reset the module
    uint16_t a = float_to_fp16(3.5f);
    uint16_t b = float_to_fp16(2.5f);
    uint16_t c = float_to_fp16(1.5f);
    uint16_t d = float_to_fp16(0.5f);
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    top->a = a;
    top->b = b;
    top->store = 1;
    clockTick();
    top->store = 0;
    clockTick();
    top->a = c;
    top->b = d;
    clockTick();

    EXPECT_EQ(top->pooled_value, a);
    
}

TEST_F(PoolingTestbench, B_max)
{
    // Reset the module
    uint16_t a = float_to_fp16(1.5f);
    uint16_t b = float_to_fp16(3.5f);
    uint16_t c = float_to_fp16(2.5f);
    uint16_t d = float_to_fp16(0.5f);
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    top->a = a;
    top->b = b;
    top->store = 1;
    clockTick();
    top->store = 0;
    clockTick();
    top->a = c;
    top->b = d;
    clockTick();

    EXPECT_EQ(top->pooled_value, b);
    
}

TEST_F(PoolingTestbench, C_max)
{
    // Reset the module
    uint16_t a = float_to_fp16(1.5f);
    uint16_t b = float_to_fp16(2.5f);
    uint16_t c = float_to_fp16(3.5f);
    uint16_t d = float_to_fp16(0.5f);
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    top->a = a;
    top->b = b;
    top->store = 1;
    clockTick();
    top->store = 0;
    clockTick();
    top->a = c;
    top->b = d;
    clockTick();

    EXPECT_EQ(top->pooled_value, c);
    
}

TEST_F(PoolingTestbench, D_max)
{
    // Reset the module
    uint16_t a = float_to_fp16(1.5f);
    uint16_t b = float_to_fp16(2.5f);
    uint16_t c = float_to_fp16(3.5f);
    uint16_t d = float_to_fp16(4.5f);
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    top->a = a;
    top->b = b;
    top->store = 1;
    clockTick();
    top->store = 0;
    clockTick();
    top->a = c;
    top->b = d;
    clockTick();

    EXPECT_EQ(top->pooled_value, d);
    
}

TEST_F(PoolingTestbench, Same)
{
    // Reset the module
    uint16_t a = float_to_fp16(0.5f);
    uint16_t b = float_to_fp16(0.5f);
    uint16_t c = float_to_fp16(0.5f);
    uint16_t d = float_to_fp16(0.5f);
    top->rst = 1;
    clockTick(2);
    top->rst = 0;
    top->a = a;
    top->b = b;
    top->store = 1;
    clockTick();
    top->store = 0;
    clockTick();
    top->a = c;
    top->b = d;
    clockTick();

    EXPECT_EQ(top->pooled_value, a);
    
}


int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("pooling_2x2.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}