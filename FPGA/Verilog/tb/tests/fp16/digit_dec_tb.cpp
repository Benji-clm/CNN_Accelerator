#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class TopReductionTestbench : public BaseTestbench
{
protected:
  void
  initializeInputs () override
  {
    top->clk = 0;
    top->rst = 1;
    top->valid_in = 0;
    clockTick ();
    top->rst = 0;
    clockTick ();
  }

  void
  clockTick ()
  {
    top->clk = 0;
    top->eval ();
    tfp->dump (ticks++);

    top->clk = 1;
    top->eval ();
    tfp->dump (ticks++);

    top->clk = 0;
  }

  // Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern
  uint16_t
  float_to_fp16 (float value)
  {
    // This is a simple conversion for testbench purposes.
    // For more accurate conversion, use a library or hardware implementation.
    union
    {
      float f;
      uint32_t u;
    } v = { value };
    uint32_t f = v.u;
    uint32_t sign = (f >> 31) & 0x1;
    int32_t exp = ((f >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = (f >> 13) & 0x3FF;
    if (exp <= 0)
      {
        exp = 0;
        frac = 0;
      }
    else if (exp >= 31)
      {
        exp = 31;
        frac = 0;
      }
    return (sign << 15) | ((exp & 0x1F) << 10) | frac;
  }

  // Helper: Convert FP16 bit pattern to float (approximate)
  float
  fp16_to_float (uint16_t h)
  {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0)
      {
        if (frac == 0)
          {
            f = sign << 31;
          }
        else
          {
            // subnormal
            exp = 127 - 15 + 1;
            while ((frac & 0x400) == 0)
              {
                frac <<= 1;
                exp--;
              }
            frac &= 0x3FF;
            f = (sign << 31) | (exp << 23) | (frac << 13);
          }
      }
    else if (exp == 31)
      {
        f = (sign << 31) | (0xFF << 23);
      }
    else
      {
        f = (sign << 31) | ((exp - 15 + 127) << 23) | (frac << 13);
      }
    union
    {
      uint32_t u;
      float f;
    } v = { f };
    return v.f;
  }

  void
  drive_sums (const float columnTop[10])
  {
    uint16_t floats_sums[10];
    for (int i = 0; i < 10; ++i)
      {
        floats_sums[i] = float_to_fp16 (columnTop[i]);
      }
    for (int i = 0; i < 10; ++i)
      {
        top->in_sum[i] = floats_sums[i];
      }
  }
};

TEST_F (TopReductionTestbench, TenParallelMatrices)
{

  top->rst = 0;
  // expecting first column of all 2x2 matrices in first clock cycle
  const float col0_r0[10]
      = { 1.1f, 1.2f, 0.4f, -4.0f, 50.0f, 0.0f, -10.2f, 8.2f, 1.3f, 7.6f };
  drive_sums (col0_r0);
  top->valid_in = 1;
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();

  float result = fp16_to_float (top->max);
  EXPECT_EQ (top->index, 4);
  EXPECT_EQ (top->valid_out, 1);
  EXPECT_NEAR (result, 50.0f, 0.01f);

  clockTick ();
  clockTick ();
}

int
main (int argc, char **argv)
{
  top = new Vdut;
  tfp = new VerilatedVcdC;

  Verilated::traceEverOn (true);
  top->trace (tfp, 99);
  tfp->open ("waveform.vcd");

  testing::InitGoogleTest (&argc, argv);
  int res = RUN_ALL_TESTS ();

  top->final ();
  tfp->close ();

  delete top;
  delete tfp;

  return res;
}
