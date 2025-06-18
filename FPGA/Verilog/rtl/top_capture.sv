/**
 * @file top_capture.sv
 * @brief Top-level module for CNN processing and BRAM capture.
 * @author Benji-clm
 * @date 2025-06-18
 *
 * @details
 * This module orchestrates the full two-layer CNN pipeline and uses a state
 * machine to arbitrate BRAM access for capturing feature maps from each layer.
 *
 * Revision 2: Corrected a critical synthesis bug. Previously, only the valid
 * signal from the first of four parallel pooling layers was used. This was
 * fixed by adding a proper AND-reduction on all four valid signals, ensuring
 * the logic for all channels is preserved during synthesis.
 */
module top_capture #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE_L1 = 5,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    parameter CONV_OUTPUT = 16,
    parameter IMAGE_SIZE = 28,
    // Parameters for the multi-dimensional input from conv_top_test
    parameter OUTPUT_CHANNELS_L1 = 4,
    parameter OUTPUT_COL_SIZE_L1 = 24,


    // layer twoooooo
    parameter KERNEL_SIZE_L2 = 3,
    parameter OUTPUT_CHANNELS_L2 = 8,
    parameter INPUT_COL_SIZE_L2 = 12,


    localparam KERNEL_SIZE_L3 = 4,
    localparam INPUT_COL_SIZE_L3 = 5,
    localparam NUM_CHANNELS_L3 = 10,
    localparam INPUT_CHANNEL_NUMBER_L3 = 8

)(
    input logic             out_stream_aclk,
    input logic             periph_resetn,
    input logic             start,

    // BRAM I/O for the local BRAM where results are written
    output logic [11:0]     bram_addr_a,
    output logic [255:0]    bram_wrdata_a,
    output logic            bram_we_a,
    output logic            write_done, // This is the main status output

    // BRAM I/O for the PS BRAM where the input image is read from
    input [255:0]			bram_rddata_a_ps,
    output [11:0]			bram_addr_a_ps,

    output logic            unused_channel_debug_out,
    output logic [DATA_WIDTH-1:0] max,
    output logic [$clog2(10)-1:0] index
);

//================================================================
// 1. CNN Pipeline Signals
//================================================================
logic [DATA_WIDTH-1:0] conv_data_out_l1 [OUTPUT_CHANNELS_L1-1:0][OUTPUT_COL_SIZE_L1-1:0];
logic                  valid_out_from_conv_1;
logic                  conv_done_from_conv_1; // Assumed done signal from L1 conv
logic [11:0]           conv_bram_addr_internal;

logic [DATA_WIDTH-1:0] relu_data_out_L1 [OUTPUT_CHANNELS_L1-1:0][OUTPUT_COL_SIZE_L1-1:0];
logic [DATA_WIDTH-1:0] pooling_data_out_L1 [OUTPUT_CHANNELS_L1-1:0][(OUTPUT_COL_SIZE_L1/2)-1:0];
logic                  pooling_valid_out_L1;

logic [DATA_WIDTH-1:0] conv_data_out_l2 [OUTPUT_CHANNELS_L2-1:0][(INPUT_COL_SIZE_L2 - KERNEL_SIZE_L2):0];
logic                  valid_out_from_conv_2;

//================================================================
// 2. BRAM Arbitration FSM
//================================================================
typedef enum logic [1:0] { IDLE, CAPTURE_L1, CAPTURE_L2, FINISHED } state_t;
state_t current_state, next_state;

// Internal wires for each capture module's BRAM interface
logic [11:0]  bram_addr_a_l1, bram_addr_a_l2;
logic [255:0] bram_wrdata_a_l1, bram_wrdata_a_l2;
logic         bram_we_a_l1, bram_we_a_l2;
logic         capture_is_done_1, capture_is_done_2;

// FSM Sequential Logic
always_ff @(posedge out_stream_aclk or negedge periph_resetn) begin
    if (!periph_resetn)
        current_state <= IDLE;
    else
        current_state <= next_state;
end

// FSM Combinational Logic (Transitions)
always_comb begin
    next_state = current_state;
    case (current_state)
        IDLE:
            if (start) next_state = CAPTURE_L1;
        CAPTURE_L1:
            if (capture_is_done_1) next_state = CAPTURE_L2;
        CAPTURE_L2:
            if (capture_is_done_2) next_state = FINISHED;
        FINISHED:
            next_state = FINISHED; // Stay in finished state
    endcase
end

// BRAM Port Multiplexer based on FSM state
always_comb begin
    case (current_state)
        CAPTURE_L1: begin
            bram_addr_a   = bram_addr_a_l1;
            bram_wrdata_a = bram_wrdata_a_l1;
            bram_we_a     = bram_we_a_l1;
        end
        CAPTURE_L2: begin
            bram_addr_a   = bram_addr_a_l2;
            bram_wrdata_a = bram_wrdata_a_l2;
            bram_we_a     = bram_we_a_l2;
        end
        default: begin // IDLE and FINISHED states
            bram_addr_a   = '0;
            bram_wrdata_a = '0;
            bram_we_a     = 1'b0;
        end
    endcase
end

assign write_done = (current_state == FINISHED);

//================================================================
// 3. CNN Pipeline Instantiation
//================================================================

// Address is only driven by the first layer
assign bram_addr_a_ps = conv_bram_addr_internal;

conv_top_test #(
    .DATA_WIDTH(DATA_WIDTH),
    .IMAGE_SIZE(IMAGE_SIZE),
    .KERNEL_SIZE(KERNEL_SIZE_L1),
    .STRIDE(STRIDE),
    .PADDING(PADDING),
    .OUTPUT_CHANNELS(OUTPUT_CHANNELS_L1),
    .OUTPUT_COL_SIZE(OUTPUT_COL_SIZE_L1)
) conv_top_inst (
    .clk(out_stream_aclk), .rst(!periph_resetn), .start(start),
    .done(conv_done_from_conv_1), .valid_out(valid_out_from_conv_1),
    .data_out(bram_rddata_a_ps), .bram_addr_a_ps(conv_bram_addr_internal),
    .data_out_x(conv_data_out_l1)
);

genvar i;
generate
    for (i = 0; i < OUTPUT_CHANNELS_L1; i = i + 1) begin : relu_channel_gen
        ReLU_column #(.COLUMN_SIZE(OUTPUT_COL_SIZE_L1)) relu_inst (
            .data_in(conv_data_out_l1[i]), .data_out(relu_data_out_L1[i])
        );
    end
endgenerate

logic pooling_valid_out_wires_L1 [OUTPUT_CHANNELS_L1-1:0];
generate
    for (i = 0; i < OUTPUT_CHANNELS_L1; i = i + 1) begin : pooling_gen
        pooling_layer #(.WINDOWS(OUTPUT_COL_SIZE_L1 / 2)) pool_inst (
            .clk(out_stream_aclk), .rst(!periph_resetn), .valid_in(valid_out_from_conv_1),
            .input_column(relu_data_out_L1[i]), .valid_out(pooling_valid_out_wires_L1[i]),
            .output_column(pooling_data_out_L1[i])
        );
    end
endgenerate

// **FIXED**: The entire pooling stage is only valid when ALL parallel channels are valid.
// This ensures that the logic for all four channels is preserved during synthesis.
always_comb begin
    logic v_out_reduced = 1'b1;
    for (int i = 0; i < OUTPUT_CHANNELS_L1; i++) begin
        v_out_reduced = v_out_reduced & pooling_valid_out_wires_L1[i];
    end
    pooling_valid_out_L1 = v_out_reduced;
end

logic [DATA_WIDTH-1:0] output_columns_L2[NUM_CHANNELS-1:0][(INPUT_COL_SIZE_L2 - KERNEL_SIZE_L2 + 1) / 2 - 1:0];

conv_layer_1 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE_L2), .INPUT_COL_SIZE(INPUT_COL_SIZE_L2),
    .NUM_CHANNELS(OUTPUT_CHANNELS_L2), .INPUT_CHANNEL_NUMBER(OUTPUT_CHANNELS_L1)
) conv_layer_1_inst (
    .clk(out_stream_aclk), .rst(!periph_resetn), .valid_in(pooling_valid_out_L1),
    .input_columns(pooling_data_out_L1), .column_valid_out(valid_out_from_conv_2),
    .fm_columns(conv_data_out_l2),
    .output_columns(output_columns_L2)
    // Other outputs like pooling and valid_out are not used in this context
);

conv_layer_2 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE_L3),
    .INPUT_COL_SIZE(INPUT_COL_SIZE_L3),
    .NUM_CHANNELS(NUM_CHANNELS_L3),
    .INPUT_CHANNEL_NUMBER(INPUT_CHANNEL_NUMBER_L3)
) conv_layer_2_inst (
    .clk(out_stream_aclk),
    .rst(!periph_resetn),
    .valid_in(valid_out_from_conv_2),
    .input_columns(output_columns_L2),
    .max(max),
    .index(index)
);

//================================================================
// 4. BRAM Capture Instantiation
//================================================================

// Gate the valid signals to ensure capture modules only run when the FSM allows
logic valid_col_l1, valid_col_l2;
assign valid_col_l1 = valid_out_from_conv_1 && (current_state == CAPTURE_L1);
assign valid_col_l2 = valid_out_from_conv_2 && (current_state == CAPTURE_L2);

fmap_capture_256 #(
    .PIX_H(24), .BASE_ADDR(12'h000)
) capture_256_bits (
    .clk(out_stream_aclk), .rst(!periph_resetn),
    .valid_col(valid_col_l1), // Use gated valid signal
    .data_col(conv_data_out_l1[0]),
    .bram_addr_a(bram_addr_a_l1), .bram_wrdata_a(bram_wrdata_a_l1),
    .bram_we_a(bram_we_a_l1), .write_done(capture_is_done_1)
);

fmap_capture_256 #(
    .PIX_H(10), .BASE_ADDR(12'h020) // Ensure this base address is correct
) capture_256_bits_l2 (
    .clk(out_stream_aclk), .rst(!periph_resetn),
    .valid_col(valid_col_l2), // Use gated valid signal
    .data_col(conv_data_out_l2[0]),
    .bram_addr_a(bram_addr_a_l2), .bram_wrdata_a(bram_wrdata_a_l2),
    .bram_we_a(bram_we_a_l2), .write_done(capture_is_done_2)
);

localparam L2_COL_SIZE = INPUT_COL_SIZE_L2 - KERNEL_SIZE_L2;

always_comb begin
    logic [DATA_WIDTH-1:0] reduced_l1_val = '0;
    logic [DATA_WIDTH-1:0] reduced_l2_val = '0;

    // Reduce unused channels from Layer 1 (channels 1 to 3)
    for (int chan = 1; chan < OUTPUT_CHANNELS_L1; chan++) begin
        for (int col = 0; col < OUTPUT_COL_SIZE_L1; col++) begin
            reduced_l1_val ^= conv_data_out_l1[chan][col];
        end
    end

    // Reduce unused channels from Layer 2 (channels 1 to 7)
    for (int chan = 1; chan < OUTPUT_CHANNELS_L2; chan++) begin
        for (int col = 0; col <= L2_COL_SIZE; col++) begin
            reduced_l2_val ^= conv_data_out_l2[chan][col];
        end
    end

    // Combine the reductions and assign to the dummy output
    unused_channel_debug_out = reduced_l1_val ^ reduced_l2_val;
end


endmodule