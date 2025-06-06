/*
 *  Verifies the results of the addfp16 module, exits with a 0 on success.
 */

#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class ReductionTestbench : public BaseTestbench
{
protected:
  void
  initializeInputs () override
  {
    top->clk = 0;
    top->rst = 1;
    clockTick ();
    clockTick ();
    top->rst = 0;
    top->valid_in = 0;
    clockTick ();
    clockTick ();
  }

  // Helper method to cycle the clock
  void
  clockTick ()
  {
    top->clk = 0;
    top->eval ();
    tfp->dump (ticks++);

    top->clk = 1;
    top->eval ();
    tfp->dump (ticks++);

    top->clk = 0; // optional reset
  }

  // Helper: Convert float to IEEE 754 half-precision (FP16) bit pattern ----
  // Thanks bitchass Radaan ^-^ (should probably remove this comment before
  // submission)
  uint16_t
  float_to_fp16 (float value)
  {
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
  setColumn (uint16_t a, uint16_t b)
  {
    top->column[0] = a;
    top->column[1] = b;
  }
};

TEST_F (ReductionTestbench, NegativeColumns)
{
  uint16_t a = float_to_fp16 (-1.5f);
  uint16_t b = float_to_fp16 (2.5f);
  setColumn (a, b);

  top->valid_in = 1;
  clockTick ();

  uint16_t c = float_to_fp16 (0.0f);
  uint16_t d = float_to_fp16 (1.0f);
  setColumn (c, d);

  top->valid_in = 1;
  clockTick ();

  top->valid_in = 0;
  clockTick ();

  float result = fp16_to_float (top->sum);
  EXPECT_NEAR (result, 2.0f, 0.01f);
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
  auto res = RUN_ALL_TESTS ();

  top->final ();
  tfp->close ();

  delete top;
  delete tfp;

  return res;
}