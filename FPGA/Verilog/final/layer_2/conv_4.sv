module conv_4 #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 4
)(
    input logic clk,
    input logic rst,

    // Input data streams
    input logic [DATA_WIDTH-1:0] data_in [KERNEL_SIZE-1:0],

    // Control signals
    input logic kernel_load,
    input logic valid_in,
    input logic valid_out, // Controlled by the parent module

    // Output data
    output logic [DATA_WIDTH-1:0] data_out
);

    // Internal Registers
    logic [DATA_WIDTH-1:0] kernel_matrix [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] image_buffer  [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] result_reg;
    logic [DATA_WIDTH-1:0] conv_reg;

    // The final output is the value from the last pipeline stage.
    assign data_out = conv_reg;

    // This single process handles all register updates for robust synthesis.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            // Reset all storage elements asynchronously.
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    kernel_matrix[i][j] <= '0;
                    image_buffer[i][j]  <= '0;
                end
            end
            result_reg <= '0;
            conv_reg   <= '0;
        end
        else begin
            // --- Data Shifting ---
            // This logic is only active when the parent provides valid input data.
            if (valid_in) begin
                if (kernel_load) begin
                    // Shift kernel matrix up by one row
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        kernel_matrix[i] <= kernel_matrix[i+1];
                    end
                    // Load the new row at the bottom
                    kernel_matrix[KERNEL_SIZE-1] <= data_in;
                end else begin
                    // Shift image buffer up by one row
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        image_buffer[i] <= image_buffer[i+1];
                    end
                    // Load the new row at the bottom
                    image_buffer[KERNEL_SIZE-1] <= data_in;
                end
            end

            // --- Data Pipeline ---

            // Stage 1: Latch the result of the combinational MAC unit.
            // This happens on the cycle that valid image data is being processed.
            if (valid_in && !kernel_load) begin
                result_reg <= final_sum;
            end

            // Stage 2: Latch the final output.
            // This is controlled by the parent module's external timing signal.
            if (valid_out) begin
                conv_reg <= result_reg;
            end
        end
    end

    // --- Combinational Multiply and Accumulate (MAC) Logic ---
    // This logic is unchanged.

    logic [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] add_stage1[7:0];
    logic [DATA_WIDTH-1:0] add_stage2[3:0];
    logic [DATA_WIDTH-1:0] add_stage3[1:0];
    logic [DATA_WIDTH-1:0] final_sum;

    generate
        for (genvar k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (genvar l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                mulfp16 mul_inst (
                    .a_in(image_buffer[k][l]),
                    .b_in(kernel_matrix[k][l]),
                    .c_out(mul_results[k][l])
                );
            end
        end

        // Awful adder tree in fp 16 -> pipelining could help

        // Stage 1: 16 inputs -> 8 outputs
        addfp16 add_s1_0(.a(mul_results[0][0]), .b(mul_results[0][1]), .sum(add_stage1[0]));
        addfp16 add_s1_1(.a(mul_results[0][2]), .b(mul_results[0][3]), .sum(add_stage1[1]));
        addfp16 add_s1_2(.a(mul_results[1][0]), .b(mul_results[1][1]), .sum(add_stage1[2]));
        addfp16 add_s1_3(.a(mul_results[1][2]), .b(mul_results[1][3]), .sum(add_stage1[3]));
        addfp16 add_s1_4(.a(mul_results[2][0]), .b(mul_results[2][1]), .sum(add_stage1[4]));
        addfp16 add_s1_5(.a(mul_results[2][2]), .b(mul_results[2][3]), .sum(add_stage1[5]));
        addfp16 add_s1_6(.a(mul_results[3][0]), .b(mul_results[3][1]), .sum(add_stage1[6]));
        addfp16 add_s1_7(.a(mul_results[3][2]), .b(mul_results[3][3]), .sum(add_stage1[7]));

        // Stage 2: 8 inputs -> 4 outputs
        addfp16 add_s2_0(.a(add_stage1[0]), .b(add_stage1[1]), .sum(add_stage2[0]));
        addfp16 add_s2_1(.a(add_stage1[2]), .b(add_stage1[3]), .sum(add_stage2[1]));
        addfp16 add_s2_2(.a(add_stage1[4]), .b(add_stage1[5]), .sum(add_stage2[2]));
        addfp16 add_s2_3(.a(add_stage1[6]), .b(add_stage1[7]), .sum(add_stage2[3]));

        // Stage 3: 4 inputs -> 2 outputs
        addfp16 add_s3_0(.a(add_stage2[0]), .b(add_stage2[1]), .sum(add_stage3[0]));
        addfp16 add_s3_1(.a(add_stage2[2]), .b(add_stage2[3]), .sum(add_stage3[1]));

        // Final Stage: 2 inputs -> 1 output
        addfp16 add_final(.a(add_stage3[0]), .b(add_stage3[1]), .sum(final_sum));
    endgenerate

endmodule