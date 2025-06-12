/*
 *  Verifies the results of the easy_image module
 */

#include "base_testbench.h"
#include <array>
#include <vector>

Vdut *top;
VerilatedVcdC *tfp;
unsigned int ticks = 0;

class EasyImageTestBench : public BaseTestbench
{
protected:
  void
  initializeInputs () override
  {
    top->clk = 0;
    top->rst = 1;
    top->valid_col = 0;
    // Initialize data_col array to zero
    for (int i = 0; i < 24; i++)
      {
        top->data_col[i] = 0;
      }
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
  drive_column (const float column_data[24])
  {
    // Convert float values to FP16 and drive data_col
    for (int i = 0; i < 24; i++)
      {
        top->data_col[i] = float_to_fp16 (column_data[i]);
      }
  }

  // Helper: Convert 8-bit grayscale back to approximate float for verification
  float
  gray8_to_float (uint8_t gray)
  {
    return static_cast<float> (gray) / 255.0f;
  }
};

TEST_F (EasyImageTestBench, SingleColumnCapture)
{
  // Test capturing and outputting a single column

  // Define test column with known FP16 values
  const float test_column[24]
      = { 0.1f,   // Should convert to 0
          0.5f,   // Should convert to ~128
          1.0f,   // Should convert to ~128 (mantissa upper bits)
          2.0f,   // Should convert to 255 (or close)
          0.25f,  // Should convert to ~64
          0.75f,  // Should convert to ~192
          1.5f,   // Should convert to ~192
          3.0f,   // Should convert to 255 (clamped)
          0.125f, // Should convert to ~32
          0.875f, // Should convert to ~224
          // Fill rest with incremental values for easy verification
          0.1f, 0.2f, 0.3f, 0.4f, 0.6f, 0.7f, 0.8f, 0.9f, 1.1f, 1.2f, 1.3f,
          1.4f, 1.6f, 1.7f };

  printf ("Testing single column capture and output...\n");

  // Send the column
  drive_column (test_column);
  top->valid_col = 1;
  clockTick ();
  top->valid_col = 0;

  for (int i = 0; i < 24; i++)
    {
      std::cout << std::hex << top->x_coordinate << std::endl;
      std::cout << std::hex << top->current_gray_pixel << std::endl;
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
