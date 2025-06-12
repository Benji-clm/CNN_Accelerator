
/*
 *  Verifies the results of the fmap_capture module
 */

#include "base_testbench.h"
#include <array>
#include <vector>

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class FmapTestBench : public BaseTestbench
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
    top->valid_out = 0;
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
  drive_column (const uint16_t column[24])
  {
    for (int i = 0; i < 24; ++i)
      {
        top->data_out[i][0] = column[i];
      }
  }
};

TEST_F (FmapTestBench, test_1)
{

  const uint16_t col[24]{ 0x10, 0x20, 0x30, 0x40, 0x50, 0x60, 0x70, 0x80,
                          0x90, 0xA0, 0x10, 0x20, 0x30, 0x40, 0x50, 0x60,
                          0x70, 0x80, 0x90, 0xA0, 0x01, 0x01, 0x01, 0x01 };
  top->valid_out = 1;
  drive_column (col);

  for (int i = 0; i < 24; i++)
    {
      clockTick ();
    }

    
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