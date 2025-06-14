module top_capture #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 5,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter CONV_OUTPUT = 16,
    parameter IMAGE_SIZE = 28  
)(
    input logic             out_stream_aclk,
    input logic             periph_resetn,
    input logic             start,

    // BRAM I/O
    output logic [11:0]     bram_addr_a,
    output logic [255:0]    bram_wrdata_a,
    output logic [3:0]      bram_we_a,
    output logic            write_done,

    input [255:0]			bram_rddata_a_ps,
    output [11:0]			bram_addr_a_ps // address
    // output [3:0]			bram_we_a_ps // write enable for each byte
);

logic [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2, data_out_3;
logic [DATA_WIDTH-1:0] data_out_4, data_out_5, data_out_6, data_out_7;
logic [DATA_WIDTH-1:0] data_out_8, data_out_9, data_out_10, data_out_11;
logic [DATA_WIDTH-1:0] data_out_12, data_out_13, data_out_14, data_out_15;
logic [DATA_WIDTH-1:0] data_out_16, data_out_17, data_out_18, data_out_19;
logic [DATA_WIDTH-1:0] data_out_20, data_out_21, data_out_22, data_out_23;

logic [15:0] better_data_col [23:0];

logic valid_out;

conv_top #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) conv_top_inst (
    .clk(out_stream_aclk),
    .rst(!periph_resetn),
    .start(start),

    .valid_out(valid_out),

    .data_out(bram_rddata_a_ps), // NOTE: this is an input
    .bram_addr_a_ps(bram_addr_a_ps), // NOTE: this is an output

    .data_out_0(data_out_0),
    .data_out_1(data_out_1),
    .data_out_2(data_out_2),
    .data_out_3(data_out_3),
    .data_out_4(data_out_4),
    .data_out_5(data_out_5),
    .data_out_6(data_out_6),
    .data_out_7(data_out_7),
    .data_out_8(data_out_8),
    .data_out_9(data_out_9),
    .data_out_10(data_out_10),
    .data_out_11(data_out_11),
    .data_out_12(data_out_12),
    .data_out_13(data_out_13),
    .data_out_14(data_out_14),
    .data_out_15(data_out_15),
    .data_out_16(data_out_16),
    .data_out_17(data_out_17),
    .data_out_18(data_out_18),
    .data_out_19(data_out_19),
    .data_out_20(data_out_20),
    .data_out_21(data_out_21),
    .data_out_22(data_out_22),
    .data_out_23(data_out_23)
);

always_comb begin
    better_data_col[0]  = data_out_0;
    better_data_col[1]  = data_out_1;
    better_data_col[2]  = data_out_2;
    better_data_col[3]  = data_out_3;
    better_data_col[4]  = data_out_4;
    better_data_col[5]  = data_out_5;
    better_data_col[6]  = data_out_6;
    better_data_col[7]  = data_out_7;
    better_data_col[8]  = data_out_8;
    better_data_col[9]  = data_out_9;
    better_data_col[10] = data_out_10;
    better_data_col[11] = data_out_11;
    better_data_col[12] = data_out_12;
    better_data_col[13] = data_out_13;
    better_data_col[14] = data_out_14;
    better_data_col[15] = data_out_15;
    better_data_col[16] = data_out_16;
    better_data_col[17] = data_out_17;
    better_data_col[18] = data_out_18;
    better_data_col[19] = data_out_19;
    better_data_col[20] = data_out_20;
    better_data_col[21] = data_out_21;
    better_data_col[22] = data_out_22;
    better_data_col[23] = data_out_23;
end

fmap_capture_256 #(
    .PIX_H(24),
    .BASE_ADDR(12'h000)
) capture_256_bits (
    .clk(out_stream_aclk),
    .rst(!periph_resetn),
    .valid_col(valid_out),
    .data_col(better_data_col),
    
    // Connect to BRAM port A for writing
    .bram_addr_a(bram_addr_a),
    .bram_wrdata_a(bram_wrdata_a),
    .bram_we_a(bram_we_a),
    .write_done(write_done)
);

endmodule
