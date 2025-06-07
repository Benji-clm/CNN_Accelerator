#include "base_testbench.h"

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class TopReductionDecisionTestbench : public BaseTestbench
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
  drive_all_columns (const float columnTop[10], const float columnBot[10])
  {
    uint16_t colfp16Top[10];
    uint16_t colfp16Bot[10];
    for (int i = 0; i < 10; ++i)
      {
        colfp16Top[i] = float_to_fp16 (columnTop[i]);
        colfp16Bot[i] = float_to_fp16 (columnBot[i]);
      }
    for (int i = 0; i < 10; ++i)
      {
        top->column[i][0] = colfp16Top[i];
        top->column[i][1] = colfp16Bot[i];
      }
  }
};

TEST_F (TopReductionDecisionTestbench, TenParallelMatricesFP16)
{

  // expecting first column of all 2x2 matrices in first clock cycle
  const float col0_r0[10]
      = { 1.0f, 0.5f, 2.0f, 1.25f, 0.75f, 3.0f, 0.25f, 1.5f, 0.125f, 2.5f };
  const float col0_r1[10]
      = { 2.0f, 1.5f, 4.0f, 2.25f, 1.75f, 6.0f, 0.75f, 2.5f, 0.375f, 3.5f };
  drive_all_columns (col0_r0, col0_r1);
  top->valid_in = 1;
  clockTick ();

  const float col1_r0[10]
      = { 3.0f, 2.5f, 6.0f, 3.25f, 2.75f, 9.0f, 1.25f, 3.5f, 0.625f, 4.5f };
  const float col1_r1[10]
      = { 4.0f, 3.5f, 8.0f, 4.25f, 3.75f, 12.0f, 1.75f, 4.5f, 0.875f, 5.5f };

  drive_all_columns (col1_r0, col1_r1);
  top->valid_in = 1;
  clockTick ();
  top->valid_in = 0;
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();

  float result = fp16_to_float (top->max);
  EXPECT_NEAR (result, 30.0f, 0.001f);
  EXPECT_EQ (top->index, 5);

  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();

  // Define floating point test values for each 2x2 matrix
  //   const float matrix_values[10][4] = {
  //     { 1.0f, 2.0f, 3.0f, 4.0f },         // Matrix 0: sum = 10.0
  //     { 0.5f, 1.5f, 2.5f, 3.5f },         // Matrix 1: sum = 8.0
  //     { 2.0f, 4.0f, 6.0f, 8.0f },         // Matrix 2: sum = 20.0
  //     { 1.25f, 2.25f, 3.25f, 4.25f },     // Matrix 3: sum = 11.0
  //     { 0.75f, 1.75f, 2.75f, 3.75f },     // Matrix 4: sum = 9.0
  //     { 3.0f, 6.0f, 9.0f, 12.0f },        // Matrix 5: sum = 30.0 (maximum)
  //     { 0.25f, 0.75f, 1.25f, 1.75f },     // Matrix 6: sum = 4.0
  //     { 1.5f, 2.5f, 3.5f, 4.5f },         // Matrix 7: sum = 12.0
  //     { 0.125f, 0.375f, 0.625f, 0.875f }, // Matrix 8: sum = 2.0
  //     { 2.5f, 3.5f, 4.5f, 5.5f }          // Matrix 9: sum = 16.0
  //   };
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
