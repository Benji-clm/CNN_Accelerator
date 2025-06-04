rm -rf obj_dir Vhex_to_ieee* *.png

verilator -Wall -sv hex_to_ieee.v \
  --cc --exe hex_to_ieee_tb.cpp \
  --top-module hex_to_ieee \
  -Wno-lint

make -C obj_dir -f Vhex_to_ieee.mk Vhex_to_ieee
./obj_dir/Vhex_to_ieee


rm -rf obj_dir Vieee_to_hex* *.png

verilator -Wall -sv ieee_to_hex.v \
  --cc --exe ieee_to_hex_tb.cpp \
  --top-module ieee_to_hex \
  -Wno-lint

make -C obj_dir -f Vieee_to_hex.mk Vieee_to_hex
./obj_dir/Vieee_to_hex
