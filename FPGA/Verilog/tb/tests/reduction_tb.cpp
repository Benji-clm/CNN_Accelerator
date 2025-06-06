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

  void
  setColumn (uint16_t a, uint16_t b)
  {
    top->column[0] = a;
    top->column[1] = b;
  }
};

TEST_F (ReductionTestbench, SimpleLoad)
{
  // First pair of inputs
  setColumn (0x12, 0x34);
  top->valid_in = 1;
  clockTick (); // Capture val_1 and val_2

  // Second pair of inputs
  setColumn (0x56, 0x78);
  top->valid_in = 1;
  clockTick ();

  // Deassert valid_in
  top->valid_in = 0;
  clockTick (); // Let valid_out go high

  // Check output
  EXPECT_EQ (top->valid_out, 1);
  EXPECT_EQ (top->sum, 0x12 + 0x34 + 0x56 + 0x78); // 18 + 52 + 86 + 120 = 276
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