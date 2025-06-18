module cv4_channel #(
    parameter DATA_WIDTH           = 16,
    parameter KERNEL_SIZE          = 4,
    parameter INPUT_COL_SIZE       = 5,
    parameter INPUT_CHANNEL_NUMBER = 8,
    parameter [DATA_WIDTH-1:0] BIAS = 16'h34f1,
    localparam PARALLEL_UNITS      = INPUT_COL_SIZE - KERNEL_SIZE + 1
)(
    input logic clk,
    input logic rst,
    input logic kernel_load,
    input logic valid_in,
    input logic [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],
    input logic [DATA_WIDTH-1:0] kernel_inputs [INPUT_CHANNEL_NUMBER-1:0][KERNEL_SIZE-1:0],
    output logic [DATA_WIDTH-1:0] output_column [PARALLEL_UNITS-1:0],
    output logic valid_out
);

    // --- Sanity Check ---
    initial begin
        if (INPUT_CHANNEL_NUMBER != 8) begin
            $fatal("This module's pipelined adder tree is hardcoded for INPUT_CHANNEL_NUMBER = 8.");
        end
    end

    // --- Internal Signals ---
    logic [DATA_WIDTH-1:0] filter_outputs [INPUT_CHANNEL_NUMBER-1:0][PARALLEL_UNITS-1:0];
    logic                  filter_valids  [INPUT_CHANNEL_NUMBER-1:0];
    logic                  combined_filter_valid;

    // Pipeline Stage 1 (8 inputs -> 4 sums)
    logic [DATA_WIDTH-1:0] sum_s1_next [3:0][PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_s1_reg  [3:0][PARALLEL_UNITS-1:0];
    logic                  valid_s1_reg;

    // Pipeline Stage 2 (4 inputs -> 2 sums)
    logic [DATA_WIDTH-1:0] sum_s2_next [1:0][PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_s2_reg  [1:0][PARALLEL_UNITS-1:0];
    logic                  valid_s2_reg;

    // Pipeline Stage 3 (2 inputs -> 1 sum)
    logic [DATA_WIDTH-1:0] sum_s3_next [PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_s3_reg  [PARALLEL_UNITS-1:0];
    logic                  valid_s3_reg;

    // Pipeline Stage 4 (Add Bias)
    logic [DATA_WIDTH-1:0] sum_s4_next [PARALLEL_UNITS-1:0];


    // --- Filter Instantiation ---
    generate
        for (genvar i = 0; i < INPUT_CHANNEL_NUMBER; i++) begin : gen_filters
            cv4_filter #(
                .DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(KERNEL_SIZE), .INPUT_COL_SIZE(INPUT_COL_SIZE)
            ) u_cv4_filter (
                .clk(clk), .rst(rst), .kernel_load(kernel_load), .valid_in(valid_in),
                .input_column(input_columns[i]), .kernel_column(kernel_inputs[i]),
                .output_column(filter_outputs[i]), .valid_out(filter_valids[i])
            );
        end
    endgenerate


    // --- Pipelined Adder Tree ---

    // Combine valid signals from all filters.
    always_comb begin
        combined_filter_valid = 1'b1;
        for (int i = 0; i < INPUT_CHANNEL_NUMBER; i++) begin
            combined_filter_valid = combined_filter_valid & filter_valids[i];
        end
    end

    // Combinational Logic for Adder Stage 1
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder1_comb
            addfp16 add_s1_0 (.a(filter_outputs[0][unit]), .b(filter_outputs[1][unit]), .sum(sum_s1_next[0][unit]));
            addfp16 add_s1_1 (.a(filter_outputs[2][unit]), .b(filter_outputs[3][unit]), .sum(sum_s1_next[1][unit]));
            addfp16 add_s1_2 (.a(filter_outputs[4][unit]), .b(filter_outputs[5][unit]), .sum(sum_s1_next[2][unit]));
            addfp16 add_s1_3 (.a(filter_outputs[6][unit]), .b(filter_outputs[7][unit]), .sum(sum_s1_next[3][unit]));
        end
    endgenerate

    // Pipeline Register for Stage 1
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s1_reg <= 1'b0;
            for (int i=0; i<4; i++) sum_s1_reg[i] <= '{default:'0};
        end else begin
            valid_s1_reg <= combined_filter_valid;
            if (combined_filter_valid) begin
                sum_s1_reg <= sum_s1_next;
            end
        end
    end

    // Combinational Logic for Adder Stage 2
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder2_comb
            addfp16 add_s2_0 (.a(sum_s1_reg[0][unit]), .b(sum_s1_reg[1][unit]), .sum(sum_s2_next[0][unit]));
            addfp16 add_s2_1 (.a(sum_s1_reg[2][unit]), .b(sum_s1_reg[3][unit]), .sum(sum_s2_next[1][unit]));
        end
    endgenerate

    // Pipeline Register for Stage 2
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s2_reg <= 1'b0;
            for (int i=0; i<2; i++) sum_s2_reg[i] <= '{default:'0};
        end else begin
            valid_s2_reg <= valid_s1_reg;
            if (valid_s1_reg) begin
                sum_s2_reg <= sum_s2_next;
            end
        end
    end

    // Combinational Logic for Adder Stage 3
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder3_comb
            addfp16 add_s3_0 (.a(sum_s2_reg[0][unit]), .b(sum_s2_reg[1][unit]), .sum(sum_s3_next[unit]));
        end
    endgenerate

    // Pipeline Register for Stage 3
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s3_reg <= 1'b0;
            sum_s3_reg   <= '{default:'0};
        end else begin
            valid_s3_reg <= valid_s2_reg;
            if (valid_s2_reg) begin
                sum_s3_reg <= sum_s3_next;
            end
        end
    end

    // Combinational Logic for Final Bias Add
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder4_comb
            addfp16 add_bias (.a(sum_s3_reg[unit]), .b(BIAS), .sum(sum_s4_next[unit]));
        end
    endgenerate

    // Final Output Register
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out     <= 1'b0;
            output_column <= '{default:'0};
        end else begin
            valid_out <= valid_s3_reg;
            if (valid_s3_reg) begin
                output_column <= sum_s4_next;
            end
        end
    end
endmodule