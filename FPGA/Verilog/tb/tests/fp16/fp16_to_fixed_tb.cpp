#include "../base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class FP16ToFixedTestbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->fp16_in = 0;
    }

    // Helper to apply input and check output
    void applyAndCheck(uint16_t fp16_in, int32_t expected_fixed)
    {
        top->fp16_in = fp16_in;
        top->eval();
        tfp->dump(ticks++);
        EXPECT_EQ(top->fixed_out, expected_fixed);
    }
};

// Helper function for FP16 encoding
uint16_t fp16_pack(bool sign, uint8_t exp, uint16_t frac) {
    return (sign << 15) | ((exp & 0x1F) << 10) | (frac & 0x3FF);
}

// Some common FP16 values
constexpr uint16_t FP16_ZERO = 0x0000;
constexpr uint16_t FP16_ONE  = 0x3C00;
constexpr uint16_t FP16_TWO  = 0x4000;
constexpr uint16_t FP16_HALF = 0x3800;
constexpr uint16_t FP16_MINUS_ONE = 0xBC00;
constexpr uint16_t FP16_MINUS_HALF = 0xB800;
constexpr uint16_t FP16_INF  = 0x7C00;
constexpr uint16_t FP16_NAN  = 0x7E00;

TEST_F(FP16ToFixedTestbench, Zero)
{
    applyAndCheck(FP16_ZERO, 0);
}

TEST_F(FP16ToFixedTestbench, One)
{
    applyAndCheck(FP16_ONE, 65536); // 1.0 * 2^16
}

TEST_F(FP16ToFixedTestbench, MinusOne)
{
    applyAndCheck(FP16_MINUS_ONE, -65536); // -1.0 * 2^16
}

TEST_F(FP16ToFixedTestbench, Two)
{
    applyAndCheck(FP16_TWO, 131072); // 2.0 * 2^16
}

TEST_F(FP16ToFixedTestbench, Half)
{
    applyAndCheck(FP16_HALF, 32768); // 0.5 * 2^16
}

TEST_F(FP16ToFixedTestbench, MinusHalf)
{
    applyAndCheck(FP16_MINUS_HALF, -32768); // -0.5 * 2^16
}

TEST_F(FP16ToFixedTestbench, SmallestNormal)
{
    uint16_t fp16_small = fp16_pack(0, 1, 0); // 2^-14
    applyAndCheck(fp16_small, 4); // 2^-14 * 2^16 = 2^2
}

TEST_F(FP16ToFixedTestbench, LargeNormal)
{
    uint16_t fp16_large = fp16_pack(0, 29, 0); // 2^14 = 16384
    applyAndCheck(fp16_large, 1073741824); // 16384 * 2^16 = 0x40000000
}

TEST_F(FP16ToFixedTestbench, Infinity)
{
    applyAndCheck(FP16_INF, 2147483647); // 0x7FFFFFFF
}

TEST_F(FP16ToFixedTestbench, MinusInfinity)
{
    applyAndCheck(0xFC00, -2147483648); // 0x80000000
}

TEST_F(FP16ToFixedTestbench, NaN)
{
    applyAndCheck(FP16_NAN, 2147483647); // 0x7FFFFFFF
}

TEST_F(FP16ToFixedTestbench, MinusNaN)
{
    applyAndCheck(0xFE00, -2147483648); // 0x80000000
}

TEST_F(FP16ToFixedTestbench, Denormal)
{
    uint16_t fp16_denorm = fp16_pack(0, 0, 1); // Denormalized number
    applyAndCheck(fp16_denorm, 0); // Treated as zero
}

TEST_F(FP16ToFixedTestbench, NegativeZero)
{
    applyAndCheck(0x8000, 0); // Negative zero treated as zero
}

int main(int argc, char **argv)
{
    top = new Vdut;
    tfp = new VerilatedVcdC;

    Verilated::traceEverOn(true);
    top->trace(tfp, 99);
    tfp->open("fp16_to_fixed_waveform.vcd");

    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    top->final();
    tfp->close();

    delete top;
    delete tfp;

    return res;
}