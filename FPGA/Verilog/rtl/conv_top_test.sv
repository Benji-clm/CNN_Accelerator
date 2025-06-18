module conv_top_test #(
    parameter DATA_WIDTH      = 16,    // Half-precision float width
    parameter KERNEL_SIZE     = 5,
    parameter STRIDE          = 1,
    parameter PADDING         = 1,
    localparam DATA_ARRAY     = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT     = 16,   // Changed to match DATA_WIDTH for FP16
    parameter IMAGE_SIZE      = 28,
    // Parameters for the multi-dimensional output
    parameter OUTPUT_CHANNELS = 4,
    parameter OUTPUT_COL_SIZE = 24
)(
    input logic clk,
    input logic rst,

    // BRAM_PS Interface
    input logic [255:0] data_out,
    // This is now driven ONLY by the master instance (edge_detect)
    output logic [11:0] bram_addr_a_ps,

    input logic start,
    output logic done,
    // New multi-dimensional output port for [channels][column_size]
    output logic [DATA_WIDTH-1:0] data_out_x [OUTPUT_CHANNELS-1:0][OUTPUT_COL_SIZE-1:0],
    output logic valid_out,
    output logic keep_top
);

// --- Internal Wires ---

// Wires to hold the outputs of each channel's convolution result
logic [DATA_WIDTH-1:0] out_0[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_1[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_2[OUTPUT_COL_SIZE-1:0];
logic [DATA_WIDTH-1:0] out_3[OUTPUT_COL_SIZE-1:0];

// Wires to gather the status signals from all four instances
logic done_wires[OUTPUT_CHANNELS-1:0];
logic valid_wires[OUTPUT_CHANNELS-1:0];


// --- Channel Instantiation ---

// Instance 0: The "Master"
// This instance drives the BRAM address and other primary control signals.
edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_master (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out),
    .data_out_x(out_0),
    .addr(bram_addr_a_ps), // Master drives the address bus
    .done(done_wires[0]),
    .valid_out_col(valid_wires[0]),
    .keep(keep_top),
    // Unconnected ports
    .write_enable(),
    .read_enable(),
    .out_col_num()
);

// Instance 1: "Slave"
// Receives the same data but does not drive the address bus.
edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_slave_1 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out),
    .data_out_x(out_1),
    .done(done_wires[1]),
    .valid_out_col(valid_wires[1]),
    // Explicitly unconnected ports for slave instances
    .addr(),
    .keep(),
    .write_enable(),
    .read_enable(),
    .out_col_num()
);

// Instance 2: "Slave"
edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_slave_2 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out),
    .data_out_x(out_2),
    .done(done_wires[2]),
    .valid_out_col(valid_wires[2]),
    // Explicitly unconnected ports
    .addr(),
    .keep(),
    .write_enable(),
    .read_enable(),
    .out_col_num()
);

// Instance 3: "Slave"
edge_detection_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .CONV_OUTPUT(CONV_OUTPUT),
    .IMAGE_SIZE(IMAGE_SIZE)
) edge_detect_slave_3 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_in(data_out),
    .data_out_x(out_3),
    .done(done_wires[3]),
    .valid_out_col(valid_wires[3]),
    // Explicitly unconnected ports
    .addr(),
    .keep(),
    .write_enable(),
    .read_enable(),
    .out_col_num()
);

// --- Output Logic ---

// Assign each internal column buffer to a channel in the output array
assign data_out_x[0] = out_0;
assign data_out_x[1] = out_1;
assign data_out_x[2] = out_2;
assign data_out_x[3] = out_3;

// **FIXED**: The entire module is 'done' and its output is 'valid' only when
// ALL four parallel channels are done/valid. This logical AND-reduction
// ensures that all channels are preserved during synthesis.
assign done      = &done_wires;
assign valid_out = &valid_wires;

endmodule
