module conv_top_test #(
    parameter DATA_WIDTH = 16,    // Half-precision float width
    parameter KERNEL_SIZE = 5,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    localparam DATA_ARRAY = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT = 16,   // Changed to match DATA_WIDTH for FP16
    parameter IMAGE_SIZE = 28,
    // Parameters for the multi-dimensional output
    parameter OUTPUT_CHANNELS = 4,
    parameter OUTPUT_COL_SIZE = 24
)(
    input logic clk,
    input logic rst,

    // BRAM_PS Interface
    input logic [255:0] data_out,   
    output logic [11:0] bram_addr_a_ps,

    input logic start,
    output logic done,
    // New multi-dimensional output port for [channels][column_size]
    output logic [DATA_WIDTH-1:0] data_out_x [OUTPUT_CHANNELS-1:0][OUTPUT_COL_SIZE-1:0],
    output logic valid_out,
    output logic keep_top
);

// Internal wires for the outputs of the instantiated modules (one per channel)
logic [DATA_WIDTH-1:0] out_0[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_1[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_2[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_3[OUTPUT_COL_SIZE-1:0];
logic done_1, done_2, done_3;


edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect (
    .clk(clk),
    .rst(rst),
    .start(start),
    .write_enable(), // Unconnected - not used
    .read_enable(),  // Unconnected - not used
    .data_in(data_out[255:0]), // Input image data
    .data_out_x(out_0), // Output convolution result for channel 0
    .addr(bram_addr_a_ps), // Address output
    .done(done),
    .valid_out_col(valid_out),
    .out_col_num(),  // Unconnected - not used
    .keep(keep_top)
);

edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_1 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out[255:0]),
    .data_out_x(out_1), // Output for channel 1
    .done(done_1)
);

edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_2 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out[255:0]),
    .data_out_x(out_2), // Output for channel 2
    .done(done_2)
);

edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_3 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out[255:0]),
    .data_out_x(out_3), // Output for channel 3
    .done(done_3)
);

// Assign each internal column buffer to a channel in the output array
assign data_out_x[0] = out_0;
assign data_out_x[1] = out_1;
assign data_out_x[2] = out_2;
assign data_out_x[3] = out_3;

endmodule
