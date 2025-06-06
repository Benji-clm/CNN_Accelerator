#include "Vhex_to_ieee.h"
#include "verilated.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

struct TestStats
{
  int total = 0;
  int passed = 0;
  int failed = 0;

  void
  record_pass ()
  {
    total++;
    passed++;
  }
  void
  record_fail ()
  {
    total++;
    failed++;
  }

  void
  print_summary ()
  {
    std::cout << "\n=== Test Summary ===" << std::endl;
    std::cout << "Total tests: " << total << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;
    std::cout << "Success rate: " << (100.0 * passed / total) << "%"
              << std::endl;
  }
} stats;

// Function to convert float to raw IEEE 754 bits
uint32_t
float_to_ieee754_bits (float f)
{
  uint32_t bits;
  std::memcpy (&bits, &f, sizeof (bits));
  return bits;
}

// Function to convert your 16-bit custom format to expected value
// This depends on your specific 16-bit IEEE 754-like format
uint16_t
compute_expected_16bit (uint16_t int_input)
{
  if (int_input == 0)
    return 0;

  // Find MSB position (same as your Verilog)
  int i = 15;
  while (i > 0 && ((int_input >> i) & 1) == 0)
    i--;

  uint32_t exponent = 15 + i;
  uint32_t mantissa;

  if (i >= 10)
    {
      // Extract 10 bits from position [i-1:i-10]
      mantissa = (int_input >> (i - 10)) & 0x3FF;

      // Check for rounding (guard bit at position i-11)
      if (i >= 11 && ((int_input >> (i - 11)) & 1))
        {
          // Check if we need to round up
          bool round_up = false;

          // Check sticky bits (bits below guard bit)
          uint32_t sticky_mask = (1u << (i - 11)) - 1;
          if ((int_input & sticky_mask) != 0)
            {
              round_up = true; // More than halfway
            }
          else
            {
              // Exactly halfway, round to even
              if (mantissa & 1)
                round_up = true;
            }

          if (round_up)
            {
              mantissa++;
              // Check for mantissa overflow
              if (mantissa > 0x3FF)
                {
                  mantissa = 0;
                  exponent++;
                }
            }
        }
    }
  else
    {
      mantissa = (int_input << (10 - i)) & 0x3FF;
    }

  if (exponent > 31)
    exponent = 31;

  return (0 << 15) | ((exponent & 0x1F) << 10) | (mantissa & 0x3FF);
}

void
test_and_compare (Vhex_to_ieee &top, auto &tick, uint16_t value)
{
  top.int_in = value;
  tick ();
  tick ();

  uint16_t actual = top.float_out;
  uint16_t expected = compute_expected_16bit (value);

  if (actual == expected)
    {
      stats.record_pass ();
    }
  else
    {
      stats.record_fail ();

      // Only print detailed breakdown for failures
      std::cout << "FAIL: int_in = " << std::dec << std::setw (5) << value
                << " -> actual = 0x" << std::hex << std::setw (4)
                << std::setfill ('0') << actual << ", expected = 0x"
                << std::setw (4) << std::setfill ('0') << expected
                << std::endl;

      std::cout << "  Actual:   S=" << ((actual >> 15) & 1)
                << " E=" << std::dec << ((actual >> 10) & 0x1F) << " M=0x"
                << std::hex << (actual & 0x3FF) << std::endl;
      std::cout << "  Expected: S=" << ((expected >> 15) & 1)
                << " E=" << std::dec << ((expected >> 10) & 0x1F) << " M=0x"
                << std::hex << (expected & 0x3FF) << std::endl;
    }
}

int
main (int argc, char **argv)
{
  Verilated::commandArgs (argc, argv);
  Vhex_to_ieee top;

  // Initialize
  top.clk = 0;
  top.rst = 1;
  top.int_in = 0;

  auto tick = [&] () {
    top.clk = !top.clk;
    top.eval ();
  };

  // Reset
  tick ();
  tick ();
  top.rst = 0;

  // Basic cases
  test_and_compare (top, tick, 0);
  test_and_compare (top, tick, 1);
  test_and_compare (top, tick, 2);
  test_and_compare (top, tick, 4);
  test_and_compare (top, tick, 8);

  // Powers of 2
  test_and_compare (top, tick, 16);
  test_and_compare (top, tick, 32);
  test_and_compare (top, tick, 64);
  test_and_compare (top, tick, 128);
  test_and_compare (top, tick, 256);
  test_and_compare (top, tick, 512);
  test_and_compare (top, tick, 1024);

  // Non-powers of 2
  test_and_compare (top, tick, 3);
  test_and_compare (top, tick, 5);
  test_and_compare (top, tick, 13);
  test_and_compare (top, tick, 255);
  test_and_compare (top, tick, 1023);

  // Large values
  test_and_compare (top, tick, 2047);
  test_and_compare (top, tick, 4095);
  test_and_compare (top, tick, 8191);
  test_and_compare (top, tick, 16383);
  test_and_compare (top, tick, 32767);
  test_and_compare (top, tick, 65535);

  stats.print_summary ();

  return 0;
}