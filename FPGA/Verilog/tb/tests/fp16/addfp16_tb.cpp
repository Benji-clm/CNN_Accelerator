/*
 *  Verifies the results of the addfp16 module, exits with a 0 on success.
 */

#include "../base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class AddFP16Testbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->a = 0;
        top->b = 0;
        // Output: sum
    }

    // Helper method to cycle the clock
    void clockTick() {
        top->eval();
        tfp->dump(ticks++);
    }

    // Helper to set inputs and evaluate
    void setInputs(uint16_t a, uint16_t b) {
        top->a = a;
        top->b = b;
        clockTick();
    }
};

// Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern
uint16_t float_to_fp16(float value) {
    // This is a simple conversion for testbench purposes.
    // For more accurate conversion, use a library or hardware implementation.
    union { float f; uint32_t u; } v = { value };
    uint32_t f = v.u;
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f >> 13) & 0x3FF;
    if (exp <= 0) {
        exp = 0;
        frac = 0;
    } else if (exp >= 31) {
        exp = 31;
        frac = 0;
    }
    return (sign << 15) | ((exp & 0x1F) << 10) | frac;
}

// Helper: Convert FP16 bit pattern to float (approximate)
float fp16_to_float(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) {
            f = sign << 31;
        } else {
            // subnormal
            exp = 127 - 15 + 1;
            while ((frac & 0x400) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (frac << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23);
    } else {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
    }
    union { uint32_t u; float f; } v = { f };
    return v.f;
}

TEST_F(AddFP16Testbench, SimpleAddition)
{
    // Reset the module
    setInputs(0, 0);

    // 1.0 + 2.0 = 3.0
    uint16_t a = float_to_fp16(-10.0f);
    uint16_t b = float_to_fp16(175.0f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 165.0f, 0.01f);
}

TEST_F(AddFP16Testbench, ComplexAddition)
{
    // Reset the module
    setInputs(0, 0);

    // 1.0 + 2.0 = 3.0
    uint16_t a = float_to_fp16(4.4453125f);
    uint16_t b = float_to_fp16(2.125f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 6.5703125f, 0.01f);
}

TEST_F(AddFP16Testbench, ComplexNegAddition)
{
    // Reset the module
    setInputs(0, 0);

    // 1.0 + 2.0 = 3.0
    uint16_t a = float_to_fp16(4.4453125f);
    uint16_t b = float_to_fp16(-2.125f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 2.3203125f, 0.01f);
}

TEST_F(AddFP16Testbench, NegativeAddition)
{
    // -1.5 + 2.5 = 1.0
    uint16_t a = float_to_fp16(-1.5f);
    uint16_t b = float_to_fp16(2.5f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 1.0f, 0.01f);
}

TEST_F(AddFP16Testbench, ZeroAddition)
{
    // 0.0 + 0.0 = 0.0
    uint16_t a = float_to_fp16(0.0f);
    uint16_t b = float_to_fp16(0.0f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 0.0f, 0.001f);
}

// TEST_F(AddFP16Testbench, OverflowToInf)
// {
//     // Large value + large value = Inf
//     uint16_t a = float_to_fp16(65504.0f); // max FP16
//     uint16_t b = float_to_fp16(65504.0f);
//     setInputs(a, b);

//     // FP16 Inf: 0x7C00
//     EXPECT_EQ(top->sum, 0x7C00);
// }

TEST_F(AddFP16Testbench, UnderflowToZero)
{
    // Smallest positive + negative = 0
    uint16_t a = float_to_fp16(1e-8f);
    uint16_t b = float_to_fp16(-1e-8f);
    setInputs(a, b);

    float result = fp16_to_float(top->sum);
    EXPECT_NEAR(result, 0.0f, 0.001f);
}


int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("addfp16_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}