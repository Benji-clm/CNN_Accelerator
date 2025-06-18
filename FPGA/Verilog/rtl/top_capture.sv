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

    // BRAM I/O for the *local* BRAM where results are written
    output logic [11:0]     bram_addr_a,
    output logic [255:0]    bram_wrdata_a,
    output logic            bram_we_a,
    output logic            write_done, // This is the main status output

    // BRAM I/O for the *PS* BRAM where the input image is read from
    input [255:0]			bram_rddata_a_ps,
    output [11:0]			bram_addr_a_ps
);

// Internal wires to connect the two sub-modules
logic [DATA_WIDTH-1:0] data_out_0, data_out_1, data_out_2, data_out_3;
logic [DATA_WIDTH-1:0] data_out_4, data_out_5, data_out_6, data_out_7;
logic [DATA_WIDTH-1:0] data_out_8, data_out_9, data_out_10, data_out_11;
logic [DATA_WIDTH-1:0] data_out_12, data_out_13, data_out_14, data_out_15;
logic [DATA_WIDTH-1:0] data_out_16, data_out_17, data_out_18, data_out_19;
logic [DATA_WIDTH-1:0] data_out_20, data_out_21, data_out_22, data_out_23;

logic [15:0]           data_col_packed [23:0];
logic                  valid_out_from_conv;
logic                  conv_done_from_conv;
logic [11:0]           conv_bram_addr_internal;
logic                  capture_is_done;


// STEP 1: Instantiate the convolution core
// It reads from the PS BRAM and outputs a column of feature map pixels
conv_top_test #(
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

    .done(conv_done_from_conv),
    .valid_out(valid_out_from_conv),

    // Connections to PS BRAM (for reading input image)
    .data_out(bram_rddata_a_ps),
    .bram_addr_a_ps(conv_bram_addr_internal),

    // Raw feature map outputs
    .data_out_0(data_out_0), .data_out_1(data_out_1), .data_out_2(data_out_2),
    .data_out_3(data_out_3), .data_out_4(data_out_4), .data_out_5(data_out_5),
    .data_out_6(data_out_6), .data_out_7(data_out_7), .data_out_8(data_out_8),
    .data_out_9(data_out_9), .data_out_10(data_out_10), .data_out_11(data_out_11),
    .data_out_12(data_out_12), .data_out_13(data_out_13), .data_out_14(data_out_14),
    .data_out_15(data_out_15), .data_out_16(data_out_16), .data_out_17(data_out_17),
    .data_out_18(data_out_18), .data_out_19(data_out_19), .data_out_20(data_out_20),
    .data_out_21(data_out_21), .data_out_22(data_out_22), .data_out_23(data_out_23)
);

// Address conversion: The convolution module might use a different address scheme
// than the BRAM. This handles the conversion.
assign bram_addr_a_ps = conv_bram_addr_internal;

// Pack the 24 individual 16-bit outputs into a single array for the capture module
// This uses a concise assignment pattern.
assign data_col_packed = '{data_out_23, data_out_22, data_out_21, data_out_20,
                           data_out_19, data_out_18, data_out_17, data_out_16,
                           data_out_15, data_out_14, data_out_13, data_out_12,
                           data_out_11, data_out_10, data_out_9,  data_out_8,
                           data_out_7,  data_out_6,  data_out_5,  data_out_4,
                           data_out_3,  data_out_2,  data_out_1,  data_out_0};


// STEP 2: Instantiate the feature map capture module
// It takes the feature map column from conv_top and writes it to the local BRAM
fmap_capture_256 #(
    .PIX_H(24),
    .BASE_ADDR(12'h000)
) capture_256_bits (
    .clk(out_stream_aclk),
    .rst(!periph_resetn),
    .valid_col(valid_out_from_conv),
    .data_col(data_col_packed),

    .conv_done(conv_done_from_conv),
    
    // Connections to Local BRAM (for writing results)
    .bram_addr_a(bram_addr_a),
    .bram_wrdata_a(bram_wrdata_a),
    .bram_we_a(bram_we_a),
    .write_done(capture_is_done) // Capture the true done signal
);

// FIXED: The top-level 'write_done' now correctly reflects when the BRAM write is finished.
assign write_done = capture_is_done;

endmodule
