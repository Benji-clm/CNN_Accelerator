module cv3_channel #(
    parameter DATA_WIDTH              = 16,
    parameter KERNEL_SIZE             = 3,
    parameter INPUT_COL_SIZE          = 12,
    parameter INPUT_CHANNEL_NUMBER    = 4,
    parameter [DATA_WIDTH-1:0] BIAS   = 16'h0000,
    localparam PARALLEL_UNITS         = INPUT_COL_SIZE - KERNEL_SIZE + 1
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

    // Internal Signals
    logic [DATA_WIDTH-1:0] filter_outputs [INPUT_CHANNEL_NUMBER-1:0][PARALLEL_UNITS-1:0];
    logic                  filter_valids  [INPUT_CHANNEL_NUMBER-1:0];
    logic [DATA_WIDTH-1:0] sum_s1_0_reg [PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_s1_1_reg [PARALLEL_UNITS-1:0];
    logic                  valid_s1_reg;
    logic [DATA_WIDTH-1:0] sum_s2_reg [PARALLEL_UNITS-1:0];
    logic                  valid_s2_reg;

    // Filter Instantiation
    generate
        for (genvar i = 0; i < INPUT_CHANNEL_NUMBER; i++) begin : gen_filters
            cv3_filter #(.DATA_WIDTH(DATA_WIDTH), .KERNEL_SIZE(KERNEL_SIZE), .INPUT_COL_SIZE(INPUT_COL_SIZE))
            u_cv3_filter (
                .clk(clk), .rst(rst), .kernel_load(kernel_load), .valid_in(valid_in),
                .input_column(input_columns[i]), .kernel_column(kernel_inputs[i]),
                .output_column(filter_outputs[i]), .valid_out(filter_valids[i])
            );
        end
    endgenerate

    // Pipelined Adder Tree
    logic [DATA_WIDTH-1:0] sum_s1_0_next [PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_s1_1_next [PARALLEL_UNITS-1:0];
    logic                  combined_filter_valid;

    always_comb begin
        combined_filter_valid = 1'b1;
        for (int i = 0; i < INPUT_CHANNEL_NUMBER; i++) begin
            combined_filter_valid = combined_filter_valid & filter_valids[i];
        end
    end

    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder1_comb
            addfp16 add_s1_0 (.a(filter_outputs[0][unit]), .b(filter_outputs[1][unit]), .sum(sum_s1_0_next[unit]));
            addfp16 add_s1_1 (.a(filter_outputs[2][unit]), .b(filter_outputs[3][unit]), .sum(sum_s1_1_next[unit]));
        end
    endgenerate

    // **FIXED**: Pipeline Stage 1 Register with correct structure
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s1_reg <= 1'b0;
            for (int i=0; i<PARALLEL_UNITS; i++) begin
                sum_s1_0_reg[i] <= '0;
                sum_s1_1_reg[i] <= '0;
            end
        end else begin
            valid_s1_reg <= combined_filter_valid;
            if (combined_filter_valid) begin
                sum_s1_0_reg <= sum_s1_0_next;
                sum_s1_1_reg <= sum_s1_1_next;
            end
        end
    end

    logic [DATA_WIDTH-1:0] sum_s2_next [PARALLEL_UNITS-1:0];
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder2_comb
            addfp16 add_s2_0 (.a(sum_s1_0_reg[unit]), .b(sum_s1_1_reg[unit]), .sum(sum_s2_next[unit]));
        end
    endgenerate

    // **FIXED**: Pipeline Stage 2 Register with correct structure
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_s2_reg <= 1'b0;
            for (int i=0; i<PARALLEL_UNITS; i++) begin
                sum_s2_reg[i] <= '0;
            end
        end else begin
            valid_s2_reg <= valid_s1_reg;
            if (valid_s1_reg) begin
                sum_s2_reg <= sum_s2_next;
            end
        end
    end

    logic [DATA_WIDTH-1:0] sum_s3_next [PARALLEL_UNITS-1:0];
    generate
        for (genvar unit = 0; unit < PARALLEL_UNITS; unit++) begin : gen_adder3_comb
            addfp16 add_bias (.a(sum_s2_reg[unit]), .b(BIAS), .sum(sum_s3_next[unit]));
        end
    endgenerate

    // **FIXED**: Final Output Register with correct structure
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            valid_out <= 1'b0;
            for (int i = 0; i < PARALLEL_UNITS; i++) begin
                output_column[i] <= '0;
            end
        end else begin
            valid_out <= valid_s2_reg;
            if (valid_s2_reg) begin
                output_column <= sum_s3_next;
            end
        end
    end
endmodule