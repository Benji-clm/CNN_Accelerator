/*
 *  Verifies the results of the pixel_generator module
 */

#include "base_testbench.h"
#include <array>
#include <iostream>
#include <vector>

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class PixelGeneratorTestBench : public BaseTestbench
{
protected:
  void
  initializeInputs () override
  {
    // Initialize all clocks
    top->out_stream_aclk = 0;
    top->s_axi_lite_aclk = 0;

    // Initialize resets (active low)
    top->axi_resetn = 0;
    top->periph_resetn = 0;

    // Initialize AXI-Lite signals
    top->s_axi_lite_araddr = 0;
    top->s_axi_lite_arvalid = 0;
    top->s_axi_lite_awaddr = 0;
    top->s_axi_lite_awvalid = 0;
    top->s_axi_lite_bready = 1;
    top->s_axi_lite_rready = 1;
    top->s_axi_lite_wdata = 0;
    top->s_axi_lite_wvalid = 0;

    // Initialize BRAM data (port A & B)
    for (int i = 0; i < 8; i++)
      {
        top->bram_rddata_a[i] = 0;
        top->bram_rddata_b[i] = 0;
      }
    // top->bram_rddata_a_ps = 0;

    // Initialize stream output
    top->out_stream_tready = 1; // Always ready to accept data

    // Apply reset
    clockTick ();
    clockTick ();

    // Release resets
    top->axi_resetn = 1;
    top->periph_resetn = 1;
    clockTick ();
    clockTick ();
  }

  void
  clockTick ()
  {
    // Tick both clocks together for simplicity
    top->out_stream_aclk = 0;
    top->s_axi_lite_aclk = 0;
    top->eval ();
#ifndef __APPLE__
    tfp->dump (ticks++);
#endif

    top->out_stream_aclk = 1;
    top->s_axi_lite_aclk = 1;
    top->eval ();
#ifndef __APPLE__
    tfp->dump (ticks++);
#endif

    top->out_stream_aclk = 0;
    top->s_axi_lite_aclk = 0;
  }

  // Helper function to write to register via AXI-Lite
  void
  writeRegister (uint32_t addr, uint32_t data)
  {
    std::cout << "Writing 0x" << std::hex << data << " to register "
              << std::dec << (addr / 4) << std::endl;

    // Write address phase
    top->s_axi_lite_awaddr = addr;
    top->s_axi_lite_awvalid = 1;
    top->s_axi_lite_wdata = data;
    top->s_axi_lite_wvalid = 1;

    clockTick ();

    // Wait for ready signals
    while (!top->s_axi_lite_awready || !top->s_axi_lite_wready)
      {
        clockTick ();
      }

    // Clear valid signals
    top->s_axi_lite_awvalid = 0;
    top->s_axi_lite_wvalid = 0;

    clockTick ();

    // Wait for response
    while (!top->s_axi_lite_bvalid)
      {
        clockTick ();
      }

    clockTick (); // One more cycle to complete
  }
};

TEST_F (PixelGeneratorTestBench, CaptureAndDisplayTest)
{
  std::cout << "Starting pixel generator test..." << std::endl;

  // Let the system settle for a few cycles
  for (int i = 0; i < 10; i++)
    {
      clockTick ();
    }

  // Start the capture process by writing 1 to register 3
  std::cout << "Setting register 3 to start capture..." << std::endl;
  top->s_axi_lite_awaddr = 0;
  top->s_axi_lite_wdata = 1;
  top->s_axi_lite_awvalid = 1;
  top->s_axi_lite_wvalid = 1;
  top->s_axi_lite_bready = 1;
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  clockTick ();
  top->s_axi_lite_awaddr = 0;
  top->s_axi_lite_wdata = 0;
  top->s_axi_lite_awvalid = 1;
  top->s_axi_lite_wvalid = 1;
  top->s_axi_lite_bready = 1;

  clockTick ();
  clockTick ();

  // Clear the write signals after one cycle
  top->s_axi_lite_awvalid = 0;
  top->s_axi_lite_wvalid = 0;

  std::cout << "Running for 3000 clock cycles..." << std::endl;

  // Just run for 3000 cycles
  for (int cycle = 0; cycle < 9000; cycle++)
    {
      clockTick ();

      if (cycle % 500 == 0)
        {
          std::cout << "Cycle " << cycle
                    << " - bram_addr_a=" << (int)top->bram_addr_a
                    << ", bram_we_a=0x" << std::hex << (int)top->bram_we_a
                    << ", bram_addr_b=" << (int)top->bram_addr_b
                    << ", tvalid=" << (int)top->out_stream_tvalid << std::dec
                    << std::endl;
        }
    }

  std::cout << "Setting register 3 to 0 (OFF) for 3000 cycles..." << std::endl;
  top->s_axi_lite_awaddr = 0;
  top->s_axi_lite_awvalid = 1;
  top->s_axi_lite_wdata = 0; // Change to 0
  top->s_axi_lite_wvalid = 1;
  top->s_axi_lite_bready = 1;

  clockTick ();

  // Clear the write signals after one cycle
  top->s_axi_lite_awvalid = 0;
  top->s_axi_lite_wvalid = 0;

  std::cout << "Running for 3000 clock cycles..." << std::endl;

  // Just run for 3000 cycles
  for (int cycle = 0; cycle < 3000000; cycle++)
    {
      clockTick ();

      if (cycle % 500 == 0)
        {
          std::cout << "Cycle " << cycle
                    << " - bram_addr_a=" << (int)top->bram_addr_a
                    << ", bram_we_a=0x" << std::hex << (int)top->bram_we_a
                    << ", bram_addr_b=" << (int)top->bram_addr_b
                    << ", tvalid=" << (int)top->out_stream_tvalid << std::dec
                    << std::endl;
        }
    }

  std::cout << "Test completed after 6000 cycles. Check waveform.vcd"
            << std::endl;
}

void
clockTick ()
{
  top->out_stream_aclk = 0;
  top->s_axi_lite_aclk = 0;
  top->eval ();
  tfp->dump (ticks++);

  top->out_stream_aclk = 1;
  top->s_axi_lite_aclk = 1;
  top->eval ();
  tfp->dump (ticks++);
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