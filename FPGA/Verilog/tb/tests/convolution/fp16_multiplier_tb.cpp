/*
 *  Verifies the results of the fp16_multiplier module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class FP16MultiplierTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->clk = 0;
        top->rst_n = 0;
        top->a = 0;
        top->b = 0;
    }

    void clockTick() {
        top->clk = 1;
        top->eval();
        tfp->dump(ticks++);

        top->clk = 0;
        top->eval();
        tfp->dump(ticks++);
    }

    // Helper to apply inputs and get result
    void applyAndCheck(uint16_t a, uint16_t b, uint16_t expected, bool expect_overflow = false, bool expect_underflow = false, bool expect_invalid = false) {
        top->a = a;
        top->b = b;
        top->rst_n = 1;
        clockTick();
        // Wait a few cycles for result to settle
        for (int i = 0; i < 2; ++i) clockTick();

        EXPECT_EQ(top->result, expected);
        EXPECT_EQ(top->overflow, expect_overflow);
        EXPECT_EQ(top->underflow, expect_underflow);
        EXPECT_EQ(top->invalid_op, expect_invalid);
    }
};

// Helper functions for FP16 encoding
uint16_t fp16_pack(bool sign, uint8_t exp, uint16_t frac) {
    return (sign << 15) | ((exp & 0x1F) << 10) | (frac & 0x3FF);
}

// Some common FP16 values
constexpr uint16_t FP16_ZERO = 0x0000;
constexpr uint16_t FP16_ONE  = 0x3C00;
constexpr uint16_t FP16_TWO  = 0x4000;
constexpr uint16_t FP16_MINUS_ONE = 0xBC00;
constexpr uint16_t FP16_INF  = 0x7C00;
constexpr uint16_t FP16_NAN  = 0x7E00;

TEST_F(FP16MultiplierTestbench, MultiplyOneByOne)
{
    // 1.0 * 1.0 = 1.0
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_ONE, FP16_ONE, FP16_ONE);
}

TEST_F(FP16MultiplierTestbench, MultiplyOneByZero)
{
    // 1.0 * 0.0 = 0.0
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_ONE, FP16_ZERO, FP16_ZERO);
}

TEST_F(FP16MultiplierTestbench, MultiplyMinusOneByTwo)
{
    // -1.0 * 2.0 = -2.0
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_MINUS_ONE, FP16_TWO, fp16_pack(1, 0x10, 0x000)); // -2.0
}

TEST_F(FP16MultiplierTestbench, MultiplyInfByZero)
{
    // inf * 0 = NaN, invalid_op
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_INF, FP16_ZERO, FP16_NAN, false, false, true);
}

TEST_F(FP16MultiplierTestbench, MultiplyInfByOne)
{
    // inf * 1 = inf
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_INF, FP16_ONE, FP16_INF);
}

TEST_F(FP16MultiplierTestbench, MultiplyNaNByOne)
{
    // NaN * 1 = NaN, invalid_op
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(FP16_NAN, FP16_ONE, FP16_NAN, false, false, true);
}

TEST_F(FP16MultiplierTestbench, MultiplyWithOverflow)
{
    // Large * Large = inf, overflow
    uint16_t large = fp16_pack(0, 0x1E, 0x3FF); // Largest normal number
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(large, large, FP16_INF, true, false, false);
}

TEST_F(FP16MultiplierTestbench, MultiplyWithUnderflow)
{
    // Small * Small = 0, underflow
    uint16_t small = fp16_pack(0, 0x01, 0x000); // Smallest normal number
    top->rst_n = 0;
    clockTick();
    top->rst_n = 1;
    clockTick();

    applyAndCheck(small, small, FP16_ZERO, false, true, false);
}

int main(int argc, char **argv)
{
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