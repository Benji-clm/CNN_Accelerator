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

  void
  drive_all_columns (const uint16_t columnTop[10],
                     const uint16_t columnBot[10])
  {
    for (int i = 0; i < 10; ++i)
      {
        top->column[i][0] = columnTop[i];
        top->column[i][1] = columnBot[i];
      }
  }
};

TEST_F (TopReductionTestbench, TenParallelMatrices)
{

  // expecting first column of all 2x2 matrices in first clock cycle
  const uint16_t col0_r0[10]
      = { 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80, 0x90, 0xA0 };
  const uint16_t col0_r1[10]
      = { 0x11, 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91, 0xA1 };
  drive_all_columns (col0_r0, col0_r1);
  top->valid_in = 1;
  clockTick ();

  const uint16_t col1_r0[10]
      = { 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x82, 0x92, 0xA2 };
  const uint16_t col1_r1[10]
      = { 0x13, 0x23, 0x33, 0x43, 0x53, 0x63, 0x73, 0x83, 0x93, 0xA3 };
  drive_all_columns (col1_r0, col1_r1);
  clockTick ();

  top->valid_in = 0;
  clockTick ();

  for (int i = 0; i < 10; ++i)
    {
      uint16_t expected = col0_r0[i] + col0_r1[i] + col1_r0[i] + col1_r1[i];

      ASSERT_EQ ((top->valid_out >> i) & 1, 1)
          << "matrix " << i << " valid flag";
      ASSERT_EQ (top->sum[i], expected) << "matrix " << i << " sum mismatch";
    }

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
