module conv_layer_2 #(
    parameter DATA_WIDTH           = 16,
    parameter KERNEL_SIZE          = 4,
    parameter INPUT_COL_SIZE       = 5,
    parameter NUM_CHANNELS         = 10,
    parameter INPUT_CHANNEL_NUMBER = 8
)(
    input logic clk,
    input logic rst,
    input logic valid_in,
    input logic [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],
    output logic [DATA_WIDTH-1:0] max,
    output logic [$clog2(NUM_CHANNELS)-1:0] index,
    output logic valid_out
);

    // --- Hardcoded Kernels and Biases (values unchanged) ---
    localparam [15:0] KERNELS[0:9][0:7][0:15] = '{...}; // Truncated for brevity
    localparam [15:0] BIASES[0:9] = '{16'h34F1, 16'h393A, 16'hAD8D, 16'hB4ED, 16'hB65F, 16'h2E6D, 16'hB3A6, 16'h3120, 16'hB010, 16'hB45A};

    // --- Internal Signals ---
    logic kernel_load_r;
    logic channel_valid_in;
    logic [DATA_WIDTH-1:0] output_columns[NUM_CHANNELS-1:0];
    typedef enum logic [1:0] {IDLE, LOAD, RUN} state_t;
    state_t state, next_state;
    logic [1:0] load_cycle_count;
    logic [DATA_WIDTH-1:0] kernel_wires [0:NUM_CHANNELS-1][0:INPUT_CHANNEL_NUMBER-1][0:KERNEL_SIZE-1];
    logic [NUM_CHANNELS-1:0] valid_out_wires;
    logic all_reduction_valid; // **FIXED**: New wire for combined valid signal

    // --- Kernel Loading State Machine (unchanged) ---
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            load_cycle_count <= '0;
        end else begin
            state <= next_state;
            if (state == LOAD) begin
                load_cycle_count <= load_cycle_count + 1;
            end
        end
    end

    always_comb begin
        next_state = state;
        kernel_load_r = 1'b0;
        case(state)
            IDLE: next_state = LOAD;
            LOAD: begin
                kernel_load_r = 1'b1;
                if (load_cycle_count == KERNEL_SIZE - 1) begin
                    next_state = RUN;
                end
            end
            RUN: next_state = RUN;
        endcase
    end

    assign channel_valid_in = valid_in | kernel_load_r;

    // --- Kernel Muxing Logic (unchanged) ---
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            for (int f = 0; f < INPUT_CHANNEL_NUMBER; f++) begin
                for (int c = 0; c < KERNEL_SIZE; c++) begin
                    kernel_wires[ch][f][c] = KERNELS[ch][f][c*KERNEL_SIZE + load_cycle_count];
                end
            end
        end
    end

    // --- Channel and Reduction Instantiation (unchanged) ---
    generate
        for (genvar ch_idx = 0; ch_idx < NUM_CHANNELS; ch_idx++) begin : gen_channel
            wire valid_out_ch;
            wire [DATA_WIDTH-1:0] column_ch [INPUT_COL_SIZE - KERNEL_SIZE:0];

            cv4_channel #(
                .DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(KERNEL_SIZE), .INPUT_COL_SIZE(INPUT_COL_SIZE),
                .INPUT_CHANNEL_NUMBER(INPUT_CHANNEL_NUMBER), .BIAS(BIASES[ch_idx])
            ) u_cv4_channel (
                .clk(clk), .rst(rst), .kernel_load(kernel_load_r), .valid_in(channel_valid_in),
                .input_columns(input_columns), .kernel_inputs(kernel_wires[ch_idx]),
                .output_column(column_ch), .valid_out(valid_out_ch)
            );

            reduction #(
                .DATA_WIDTH(DATA_WIDTH), .mat_height((INPUT_COL_SIZE - KERNEL_SIZE - 1) / 2)
            ) redu (
                .clk(clk), .rst(rst), .valid_in(valid_out_ch), .column(column_ch),
                .sum(output_columns[ch_idx]), .valid_out(valid_out_wires[ch_idx])
            );
        end
    endgenerate

    // **FIXED**: Synchronize all reduction outputs before the final decision.
    always_comb begin
        all_reduction_valid = 1'b1;
        for (int i = 0; i < NUM_CHANNELS; i++) begin
            all_reduction_valid = all_reduction_valid & valid_out_wires[i];
        end
    end

    // --- Final Digit Decision ---
    digit_dec #(
        .DATA_WIDTH(DATA_WIDTH), .N_MATS(NUM_CHANNELS)
    ) digit (
        .clk(clk), .rst(rst),
        .valid_in(all_reduction_valid), // **FIXED**: Use the synchronized valid signal
        .in_sum(output_columns),
        .max(max), .index(index), .valid_out(valid_out)
    );

endmodule