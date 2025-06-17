module cv4_channel #(
    // Parameters
    parameter DATA_WIDTH           = 16,  // Data width for each element (e.g., FP16)
    parameter KERNEL_SIZE          = 4,   // The size of the convolution kernel (e.g., 3x3)
    parameter INPUT_COL_SIZE       = 5,  // Height of the input data column
    parameter INPUT_CHANNEL_NUMBER = 8,   // Number of parallel input channels/filters
    parameter BIAS                 = 16'h34f1
)(
    input logic clk,
    input logic rst,

    // --- Control Signals (shared across all filters) ---
    input logic kernel_load,  // Assert to load kernel weights into all filters
    input logic valid_in,     // Assert when all input columns and kernels are valid

    // --- Data Inputs ---
    // Unpacked array for input columns for all channels.
    // Dims: [Channel][Column Element]
    input logic [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],

    // Unpacked array for kernel columns for all channels.
    // Dims: [Channel][Kernel Element]
    input logic [DATA_WIDTH-1:0] kernel_inputs [INPUT_CHANNEL_NUMBER-1:0][KERNEL_SIZE-1:0],

    // --- Data Outputs ---
    // The resulting output column after summation and bias addition.
    output logic [DATA_WIDTH-1:0] output_column [PARALLEL_UNITS-1:0],
    // Asserted when the output_column data is valid.
    output logic valid_out
);

    // Calculate the size of the output column based on input and kernel size.
    localparam PARALLEL_UNITS = INPUT_COL_SIZE - KERNEL_SIZE + 1;

    // --- Internal Wires and Registers ---

    // Array of wires to capture the output from each filter instance.
    logic [DATA_WIDTH-1:0] filter_out_w [INPUT_CHANNEL_NUMBER-1:0][PARALLEL_UNITS-1:0];

    // Wire to capture the valid signal from the filters. Since they are synchronous,
    // we only need to monitor one of them.
    logic filters_valid_w;
    //--------------------------------------------------------------------------
    // Instantiate cv3_filter modules using a generate block
    //--------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < INPUT_CHANNEL_NUMBER; k = k + 1) begin : gen_filters
            // Use a generate if-else for robust conditional connections
            if (k == 0) begin : first_filter_inst
                cv4_filter #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .KERNEL_SIZE(KERNEL_SIZE),
                    .INPUT_COL_SIZE(INPUT_COL_SIZE)
                ) filter_inst (
                    .clk(clk),
                    .rst(rst),
                    .kernel_load(kernel_load),
                    .valid_in(valid_in),
                    .input_column(input_columns[k]),
                    .kernel_column(kernel_inputs[k]),
                    .output_column(filter_out_w[k]),
                    .valid_out(filters_valid_w) // Connect valid_out only for the first instance
                );
            end else begin : other_filter_insts
                 cv4_filter #(
                    .DATA_WIDTH(DATA_WIDTH),
                    .KERNEL_SIZE(KERNEL_SIZE),
                    .INPUT_COL_SIZE(INPUT_COL_SIZE)
                ) filter_inst (
                    .clk(clk),
                    .rst(rst),
                    .kernel_load(kernel_load),
                    .valid_in(valid_in),
                    .input_column(input_columns[k]),
                    .kernel_column(kernel_inputs[k]),
                    .output_column(filter_out_w[k]),
                    .valid_out() // Leave unconnected for other instances
                );
            end
        end
    endgenerate


    //--------------------------------------------------------------------------
    // Combinational Adder Tree Logic
    //--------------------------------------------------------------------------
    // This block sums the outputs from the four filters and adds the bias.
    // A generate loop is used for clean, scalable vector-wise addition.
    initial begin
        if (INPUT_CHANNEL_NUMBER != 8) begin
            $fatal("This module's adder tree is designed for INPUT_CHANNEL_NUMBER = 4.");
        end
    end

    // --- Wires for the 8-input Adder Tree ---
    // Stage 1: 8 inputs -> 4 outputs
    logic [DATA_WIDTH-1:0] sum_stage1_w [3:0][PARALLEL_UNITS-1:0];
    // Stage 2: 4 inputs -> 2 outputs
    logic [DATA_WIDTH-1:0] sum_stage2_w [1:0][PARALLEL_UNITS-1:0];
    // Stage 3: 2 inputs -> 1 output
    logic [DATA_WIDTH-1:0] sum_stage3_w [PARALLEL_UNITS-1:0];

    // This wire holds the final combinational result before the pipeline register.
    logic [DATA_WIDTH-1:0] combined_result_w [PARALLEL_UNITS-1:0];

    genvar i;
    generate
        for (i = 0; i < PARALLEL_UNITS; i = i + 1) begin : fp16_adder_tree
            // --- Stage 1: Add filter outputs in pairs (8 inputs -> 4 outputs) ---
            addfp16 adder_s1_0(.a(filter_out_w[0][i]), .b(filter_out_w[1][i]), .sum(sum_stage1_w[0][i]));
            addfp16 adder_s1_1(.a(filter_out_w[2][i]), .b(filter_out_w[3][i]), .sum(sum_stage1_w[1][i]));
            addfp16 adder_s1_2(.a(filter_out_w[4][i]), .b(filter_out_w[5][i]), .sum(sum_stage1_w[2][i]));
            addfp16 adder_s1_3(.a(filter_out_w[6][i]), .b(filter_out_w[7][i]), .sum(sum_stage1_w[3][i]));

            // --- Stage 2: Add stage 1 results in pairs (4 inputs -> 2 outputs) ---
            addfp16 adder_s2_0(.a(sum_stage1_w[0][i]), .b(sum_stage1_w[1][i]), .sum(sum_stage2_w[0][i]));
            addfp16 adder_s2_1(.a(sum_stage1_w[2][i]), .b(sum_stage1_w[3][i]), .sum(sum_stage2_w[1][i]));

            // --- Stage 3: Add stage 2 results (2 inputs -> 1 output) ---
            addfp16 adder_s3  (.a(sum_stage2_w[0][i]), .b(sum_stage2_w[1][i]), .sum(sum_stage3_w[i]));

            // --- Final Stage: Add the bias to the final sum ---
            addfp16 adder_bias(.a(sum_stage3_w[i]),   .b(BIAS),                 .sum(combined_result_w[i]));
        end
    endgenerate


    //--------------------------------------------------------------------------
    // Output Pipeline Register
    //--------------------------------------------------------------------------
    // This register delays the final output by one clock cycle to allow time
    // for the adder tree, improving the module's maximum frequency (Fmax).

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < PARALLEL_UNITS; i = i + 1) begin
                output_column[i] <= '0;
            end
            valid_out     <= 1'b0;
        end else begin
            // Latch the combined result when the filters provide valid data.
            if (filters_valid_w) begin
                output_column <= combined_result_w;
            end
            // The output valid signal is a delayed version of the internal valid signal.
            valid_out <= filters_valid_w;
        end
    end

endmodule
