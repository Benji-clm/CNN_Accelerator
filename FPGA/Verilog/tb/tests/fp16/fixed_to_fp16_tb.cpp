#include "../base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class FixedToFP16Testbench : public BaseTestbench
{
protected:
    void initializeInputs() override
    {
        top->fixed_in = 0;
    }

    // Helper to apply input and check output
    void applyAndCheck(int32_t fixed_in, uint16_t expected_fp16)
    {
        top->fixed_in = fixed_in;
        top->eval();
        tfp->dump(ticks++);
        EXPECT_EQ(top->fp16_out, expected_fp16);
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

TEST_F(FixedToFP16Testbench, Zero)
{
    applyAndCheck(0, FP16_ZERO); // 0.0 -> 0x0000
}

TEST_F(FixedToFP16Testbench, One)
{
    applyAndCheck(65536, FP16_ONE); // 1.0 * 2^16 -> 0x3C00
}

TEST_F(FixedToFP16Testbench, MinusOne)
{
    applyAndCheck(-65536, FP16_MINUS_ONE); // -1.0 * 2^16 -> 0xBC00
}

TEST_F(FixedToFP16Testbench, Two)
{
    applyAndCheck(131072, FP16_TWO); // 2.0 * 2^16 -> 0x4000
}

TEST_F(FixedToFP16Testbench, Half)
{
    applyAndCheck(32768, FP16_HALF); // 0.5 * 2^16 -> 0x3800
}

TEST_F(FixedToFP16Testbench, MinusHalf)
{
    applyAndCheck(-32768, FP16_MINUS_HALF); // -0.5 * 2^16 -> 0xB800
}

TEST_F(FixedToFP16Testbench, OnePointFive)
{
    applyAndCheck(98304, 0x3E00); // 1.5 * 2^16 -> 0x3E00
}

TEST_F(FixedToFP16Testbench, Quarter)
{
    applyAndCheck(16384, 0x3400); // 0.25 * 2^16 -> 0x3400
}

TEST_F(FixedToFP16Testbench, LargePositive)
{
    applyAndCheck(160768, 0x40E8);
}

TEST_F(FixedToFP16Testbench, LargeNegative)
{
    applyAndCheck(-91136, 0xBD90);
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