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
    output logic [$clog2(10)-1:0] index,

    output logic final_valid_out
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
logic                  valid_out_from_pooling_L2;

//================================================================
// 2. BRAM Arbitration FSM
//================================================================
typedef enum logic [1:0] { IDLE, CAPTURE_L1, FINISHED } state_t;
state_t current_state, next_state;

// Internal wires for the L1 capture module's BRAM interface
logic [11:0]  bram_addr_a_l1_internal;
logic [255:0] bram_wrdata_a_l1;
logic         bram_we_a_l1;
logic         capture_is_done_1;

// **NEW**: Signals for dynamic L1 feature map capture
logic [$clog2(OUTPUT_CHANNELS_L1)-1:0] l1_capture_index;
logic [11:0]  l1_dynamic_base_addr;

// **NEW**: Counter to cycle through L1 feature maps for capture.
// This counter is ONLY triggered by the final valid output of the entire network.
always_ff @(posedge out_stream_aclk or negedge periph_resetn) begin
    if (!periph_resetn) begin
        l1_capture_index <= '0;
    end else if (current_state == IDLE) begin
        l1_capture_index <= '0; // Reset counter when FSM is idle
    end else if (final_valid_out) begin // Trigger is the valid signal from conv_layer_2
        if (l1_capture_index < OUTPUT_CHANNELS_L1 - 1) begin
            l1_capture_index <= l1_capture_index + 1;
        end
    end
end

// **NEW**: Calculate the dynamic base address for L1 capture.
// The base address is changed by an offset of 30 for each new channel captured.
assign l1_dynamic_base_addr = l1_capture_index * 30;


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
            if (capture_is_done_1) next_state = FINISHED;
        FINISHED:
            next_state = FINISHED; // Stay in finished state
    endcase
end

// BRAM Port Multiplexer based on FSM state
always_comb begin
    // Default assignments to avoid latches
    bram_addr_a   = '0;
    bram_wrdata_a = '0;
    bram_we_a     = 1'b0;

    case (current_state)
        CAPTURE_L1: begin
            // **FIXED**: Add the dynamic offset to the relative address from the capture module.
            bram_addr_a   = l1_dynamic_base_addr + bram_addr_a_l1_internal;
            bram_wrdata_a = bram_wrdata_a_l1;
            bram_we_a     = bram_we_a_l1;
        end
        default:; // For IDLE and FINISHED, use default assignments
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
    .clk(out_stream_aclk), 
    .rst(!periph_resetn), 
    .start(start),
    .done(conv_done_from_conv_1), 
    .valid_out(valid_out_from_conv_1),
    .data_out(bram_rddata_a_ps), 
    .bram_addr_a_ps(conv_bram_addr_internal),
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
            .clk(out_stream_aclk), 
            .rst(!periph_resetn), 
            .valid_in(valid_out_from_conv_1),
            .input_column(relu_data_out_L1[i]), 
            .valid_out(pooling_valid_out_wires_L1[i]),
            .output_column(pooling_data_out_L1[i])
        );
    end
endgenerate

always_comb begin
    logic v_out_reduced = 1'b1;
    for (int j = 0; j < OUTPUT_CHANNELS_L1; j++) begin
        v_out_reduced = v_out_reduced & pooling_valid_out_wires_L1[j];
    end
    pooling_valid_out_L1 = v_out_reduced;
end

logic [DATA_WIDTH-1:0] output_columns_L2[OUTPUT_CHANNELS_L2-1:0][(INPUT_COL_SIZE_L2 - KERNEL_SIZE_L2 + 1) / 2 - 1:0];

conv_layer_1 #(
    .DATA_WIDTH(DATA_WIDTH), 
    .KERNEL_SIZE(KERNEL_SIZE_L2), 
    .INPUT_COL_SIZE(INPUT_COL_SIZE_L2),
    .NUM_CHANNELS(OUTPUT_CHANNELS_L2), 
    .INPUT_CHANNEL_NUMBER(OUTPUT_CHANNELS_L1)
) conv_layer_1_inst (
    .clk(out_stream_aclk), 
    .rst(!periph_resetn), 
    .valid_in(pooling_valid_out_L1),
    .input_columns(pooling_data_out_L1), 
    .column_valid_out(valid_out_from_conv_2),
    .fm_columns(conv_data_out_l2), 
    .output_columns(output_columns_L2),
    .valid_out(valid_out_from_pooling_L2)
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
    .valid_in(valid_out_from_pooling_L2),
    .input_columns(output_columns_L2), 
    .valid_out(final_valid_out),
    .max(max), 
    .index(index)
);

//================================================================
// 4. BRAM Capture Instantiation
//================================================================

logic valid_col_l1;
assign valid_col_l1 = valid_out_from_conv_1 && (current_state == CAPTURE_L1);

fmap_capture_256 #(
    .PIX_H(24), 
    .BASE_ADDR(12'h000) // **FIXED**: Base address is 0; offset is added externally.
) capture_256_bits (
    .clk(out_stream_aclk), 
    .rst(!periph_resetn),
    .valid_col(valid_col_l1),
    .data_col(conv_data_out_l1[l1_capture_index]), // Data column selected by the counter
    .bram_addr_a(bram_addr_a_l1_internal), // Outputs a relative address
    .bram_wrdata_a(bram_wrdata_a_l1),
    .bram_we_a(bram_we_a_l1), 
    .write_done(capture_is_done_1)
);


localparam L2_COL_SIZE = INPUT_COL_SIZE_L2 - KERNEL_SIZE_L2;

always_comb begin
    logic [DATA_WIDTH-1:0] reduced_l1_val = '0;
    logic [DATA_WIDTH-1:0] reduced_l2_val = '0;

    // Reduce unused channels from Layer 1. Since the capture index changes,
    // all channels are technically used over time. This logic remains as a
    // safeguard for synthesis if not all indices are reached.
    for (int chan = 0; chan < OUTPUT_CHANNELS_L1; chan++) begin
        for (int col = 0; col < OUTPUT_COL_SIZE_L1; col++) begin
            reduced_l1_val ^= conv_data_out_l1[chan][col];
        end
    end

    // Reduce all channels from Layer 2 to prevent them from being synthesized away.
    for (int chan = 0; chan < OUTPUT_CHANNELS_L2; chan++) begin
        for (int col = 0; col <= L2_COL_SIZE; col++) begin
            reduced_l2_val ^= conv_data_out_l2[chan][col];
        end
    end

    // Combine the reductions and assign to the dummy output
    unused_channel_debug_out = reduced_l1_val ^ reduced_l2_val ^ valid_out_from_pooling_L2;
end


endmodule