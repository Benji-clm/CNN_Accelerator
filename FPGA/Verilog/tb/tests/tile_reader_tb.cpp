/*
 *  Verifies the results of the tile_reader module
 */

#include "base_testbench.h"
#include <array>
#include <iostream>
#include <vector>

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class TileReaderTestBench : public BaseTestbench
{
protected:
  void
  initializeInputs () override
  {
    top->clk = 0;
    top->rst_n = 0;
    // Initialize 256-bit wide vector (8 x 32-bit words)
    for (int i = 0; i < 8; i++)
      {
        top->bram_rdata[i] = 0;
      }
    clockTick ();
    top->rst_n = 1;
    clockTick ();
  }

  void
  clockTick ()
  {
    top->clk = 0;
    top->eval ();
#ifndef __APPLE__
    tfp->dump (ticks++);
#endif

    top->clk = 1;
    top->eval ();
#ifndef __APPLE__
    tfp->dump (ticks++);
#endif

    top->clk = 0;
  }

  // Helper: Set BRAM data based on address
  void
  setBramData (uint16_t addr, const std::vector<uint8_t> &pixels)
  {
    // Simulate BRAM read response
    // Verilator represents wide vectors as arrays of 32-bit words
    // 256 bits = 8 words of 32 bits each

    // Clear the BRAM data first
    for (int i = 0; i < 8; i++)
      {
        top->bram_rdata[i] = 0;
      }

    // Pack pixel data into the wide vector
    // Each pixel is 8 bits, so we can fit 4 pixels per 32-bit word
    for (int i = 0; i < std::min (24, (int)pixels.size ()); i++)
      {
        int word_idx = i / 4;         // Which 32-bit word
        int bit_offset = (i % 4) * 8; // Bit position within the word
        top->bram_rdata[word_idx] |= ((uint32_t)pixels[i]) << bit_offset;
      }
  }
};

TEST_F (TileReaderTestBench, SingleColumnRead)
{
  // Test reading a single column of 24 pixels
  std::vector<uint8_t> testPixels;
  for (int i = 0; i < 24; i++)
    {
      testPixels.push_back (i + 1); // Pixels 1-24
    }

  std::vector<std::vector<uint8_t>> bramData;
  bramData.push_back (testPixels);

  std::cout << "Starting single column read test..." << std::endl;

  // Provide initial BRAM data (address 0)
  setBramData (0, testPixels);

  // Run simulation for a reasonable number of cycles
  std::vector<uint8_t> capturedPixels;
  for (int cycle = 0; cycle < 100; cycle++)
    {
      // Respond to BRAM address changes
      static uint16_t lastAddr = 0xFFFF;
      if (top->bram_addr != lastAddr)
        {
          lastAddr = top->bram_addr;
          if (top->bram_addr < bramData.size ())
            {
              setBramData (top->bram_addr, bramData[top->bram_addr]);
            }
        }

      clockTick ();

      if (top->pixel_valid)
        {
          capturedPixels.push_back (top->pixel);
          std::cout << "Captured pixel " << capturedPixels.size () << ": "
                    << (int)top->pixel << std::endl;
        }

      if (capturedPixels.size () >= 24)
        break;
    }

  // Verify we got the right number of pixels
  EXPECT_EQ (capturedPixels.size (), 24);

  // Verify pixel values
  for (int i = 0; i < std::min (24, (int)capturedPixels.size ()); i++)
    {
      EXPECT_EQ (capturedPixels[i], i + 1) << "Pixel " << i << " mismatch";
    }
}

// TEST_F (TileReaderTestBench, MultipleColumnRead)
// {
//   // Test reading multiple columns (3 columns of 24 pixels each)
//   std::vector<std::vector<uint8_t>> bramData;

//   for (int col = 0; col < 3; col++)
//     {
//       std::vector<uint8_t> columnPixels;
//       for (int row = 0; row < 24; row++)
//         {
//           columnPixels.push_back ((col * 24) + row + 1);
//         }
//       bramData.push_back (columnPixels);
//     }

//   std::cout << "Starting multiple column read test..." << std::endl;

//   // Run simulation
//   std::vector<uint8_t> capturedPixels;
//   for (int cycle = 0; cycle < 200; cycle++)
//     {
//       // Respond to BRAM address changes
//       static uint16_t lastAddr = 0xFFFF;
//       if (top->bram_addr != lastAddr)
//         {
//           lastAddr = top->bram_addr;
//           std::cout << "BRAM address changed to: " << (int)top->bram_addr
//                     << std::endl;
//           if (top->bram_addr < bramData.size ())
//             {
//               setBramData (top->bram_addr, bramData[top->bram_addr]);
//             }
//         }

//       clockTick ();

//       if (top->pixel_valid)
//         {
//           capturedPixels.push_back (top->pixel);
//           std::cout << "Captured pixel " << capturedPixels.size () << ": "
//                     << (int)top->pixel << std::endl;
//         }

//       if (capturedPixels.size () >= 72)
//         break;
//     }

//   // Verify we got the right number of pixels
//   EXPECT_EQ (capturedPixels.size (), 72);

//   // Verify pixel values
//   for (int i = 0; i < std::min (72, (int)capturedPixels.size ()); i++)
//     {
//       EXPECT_EQ (capturedPixels[i], i + 1) << "Pixel " << i << " mismatch";
//     }
// }

// TEST_F (TileReaderTestBench, StateMachineTest)
// {
//   // Test the state machine transitions with a known pattern
//   std::vector<uint8_t> testPixels;
//   for (int i = 0; i < 24; i++)
//     {
//       testPixels.push_back (0xAA); // Test pattern
//     }

//   std::cout << "Starting state machine test..." << std::endl;

//   // Run simulation and observe state transitions
//   for (int cycle = 0; cycle < 50; cycle++)
//     {
//       // Respond to BRAM address changes
//       static uint16_t lastAddr = 0xFFFF;
//       if (top->bram_addr != lastAddr)
//         {
//           lastAddr = top->bram_addr;
//           setBramData (top->bram_addr, testPixels);
//           std::cout << "BRAM read at address: " << (int)top->bram_addr
//                     << std::endl;
//         }

//       clockTick ();

//       // Print state information
//       std::cout << "Cycle " << cycle
//                 << ": pixel_valid=" << (int)top->pixel_valid
//                 << ", pixel=" << (int)top->pixel
//                 << ", bram_addr=" << (int)top->bram_addr << std::endl;

//       // Stop after we see some valid pixels
//       if (cycle > 10 && top->pixel_valid)
//         {
//           EXPECT_EQ (top->pixel, 0xAA) << "Expected test pattern pixel";
//           break;
//         }
//     }
// }

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
