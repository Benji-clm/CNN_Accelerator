module cv3_channel #(
    // Parameters
    parameter DATA_WIDTH           = 16,  // Data width for each element (e.g., FP16)
    parameter KERNEL_SIZE          = 3,   // The size of the convolution kernel (e.g., 3x3)
    parameter INPUT_COL_SIZE       = 12,  // Height of the input data column
    parameter INPUT_CHANNEL_NUMBER = 4,   // Number of parallel input channels/filters
    parameter BIAS                 = 16'hb06a
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

    // Wires for the intermediate stages of the adder tree.
    // This structure helps with synthesis and clarity. Assumes INPUT_CHANNEL_NUMBER = 4.
    logic [DATA_WIDTH-1:0] sum_stage1_0_w [PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_stage1_1_w [PARALLEL_UNITS-1:0];
    logic [DATA_WIDTH-1:0] sum_stage2_w [PARALLEL_UNITS-1:0];

    // This wire holds the final combinational result before the pipeline register.
    logic [DATA_WIDTH-1:0] combined_result_w [PARALLEL_UNITS-1:0];


    //--------------------------------------------------------------------------
    // Instantiate cv3_filter modules using a generate block
    //--------------------------------------------------------------------------
    genvar k;
    generate
        for (k = 0; k < INPUT_CHANNEL_NUMBER; k = k + 1) begin : gen_filters
            // Use a generate if-else for robust conditional connections
            if (k == 0) begin : first_filter_inst
                cv3_filter #(
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
                 cv3_filter #(
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
    // Note: This adder tree is hard-coded for 4 channels.
    initial begin
        if (INPUT_CHANNEL_NUMBER != 4) begin
            $fatal("This module's adder tree is designed for INPUT_CHANNEL_NUMBER = 4.");
        end
    end

    genvar i;
    generate
        for (i = 0; i < PARALLEL_UNITS; i = i + 1) begin : fp16_adder_tree
            // Stage 1: Add filter outputs in pairs
            addfp16 adder_s1_0 (
                .a(filter_out_w[0][i]),
                .b(filter_out_w[1][i]),
                .sum(sum_stage1_0_w[i])
            );

            addfp16 adder_s1_1 (
                .a(filter_out_w[2][i]),
                .b(filter_out_w[3][i]),
                .sum(sum_stage1_1_w[i])
            );

            // Stage 2: Add the results of the first stage
            addfp16 adder_s2 (
                .a(sum_stage1_0_w[i]),
                .b(sum_stage1_1_w[i]),
                .sum(sum_stage2_w[i])
            );

            // Stage 3: Add the bias to the final sum
            addfp16 adder_bias (
                .a(sum_stage2_w[i]),
                .b(BIAS),
                .sum(combined_result_w[i])
            );
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
