#include "Vieee_to_hex.h"
#include "verilated.h"
#include <cassert>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>

int16_t
ieee754_half_to_int (uint16_t half_float)
{
  // Extract components
  uint16_t sign = (half_float >> 15) & 0x1;
  uint16_t exponent = (half_float >> 10) & 0x1F;
  uint16_t mantissa = half_float & 0x3FF;

  if (exponent == 0)
    {
      return 0;
    }

  if (exponent == 0x1F)
    {
      return 0;
    }

  int actual_exp = exponent - 15;

  if (actual_exp < 0)
    {
      return 0;
    }

  uint32_t significand = (1 << 10) | mantissa;

  int32_t result;
  if (actual_exp >= 10)
    {
      result = significand << (actual_exp - 10);
    }
  else
    {
      result = significand >> (10 - actual_exp);
    }

  if (sign)
    {
      result = -result;
    }

  if (result > INT16_MAX)
    return INT16_MAX;
  if (result < INT16_MIN)
    return INT16_MIN;

  return static_cast<int16_t> (result);
}

int
main (int argc, char **argv)
{
  Verilated::commandArgs (argc, argv);
  Vieee_to_hex top;

  // Initialize
  top.clk = 0;
  top.rst = 1;
  top.float_in = 0;

  auto tick = [&] () {
    top.clk = !top.clk;
    top.eval ();
  };

  // Reset
  tick ();
  tick ();
  top.rst = 0;

  // Test some values
  uint16_t test_values[] = { 0x3C00, // 1.0
                             0x4000, // 2.0
                             0x4200, // 3.0
                             0x3800, // 0.5
                             0x7000, 0x7F6F };

  for (auto val : test_values)
    {
      top.float_in = val;
      tick ();
      tick ();

      int16_t expected = ieee754_half_to_int (val);
      std::cout << "Input: 0x" << std::hex << val << " Expected: " << std::dec
                << expected << " Got: " << top.int_out << std::endl;
    }

  return 0;
}