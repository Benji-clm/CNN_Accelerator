module conv_3 #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3
)(
    input logic clk,
    input logic rst,

    // Input data streams
    input logic signed [DATA_WIDTH-1:0] data_in [KERNEL_SIZE - 1: 0],

    // Control signals
    input logic kernel_load, // High to load kernel weights, low to process image data
    input logic valid_in,    // High when input data is valid
    input logic valid_out,  

    // Output data
    output logic signed [DATA_WIDTH-1:0] data_out
);

    // Internal Registers
    logic signed [DATA_WIDTH-1:0] kernel_matrix [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] image_buffer  [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic signed [DATA_WIDTH-1:0] conv_reg;

    localparam signed [DATA_WIDTH*2-1:0] ROUND_CONSTANT = 1 << (14 - 1);
    // The final output is the value from the last pipeline stage.
    assign data_out = conv_reg;

    // This single process handles all register updates.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    kernel_matrix[i][j] <= '0;
                    image_buffer[i][j]  <= '0;
                end
            end
            conv_reg   <= '0;
        end
        else begin
            // --- Data Shifting ---
            // This logic is only active when the parent provides valid input data.
            if (valid_in) begin
                if (kernel_load) begin
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        kernel_matrix[i] <= kernel_matrix[i+1];
                    end
                    kernel_matrix[KERNEL_SIZE-1] <= data_in;
                end else begin
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        image_buffer[i] <= image_buffer[i+1];
                    end
                    image_buffer[KERNEL_SIZE-1] <= data_in;
                end
            end

            // --- Data Pipeline ---
            // Stage 2: Latch the final output.
            // This is controlled by the parent module's external timing signal.
            if (valid_out) begin
                conv_reg <= final_sum;
            end
        end
    end

    // --- Combinational Multiply and Accumulate (MAC) Logic for Q2.14---
    // This logic performs fixed-point arithmetic for the Q2.14 data format.

    // Intermediate wires for multiplication results
    logic signed [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic signed [DATA_WIDTH*2-1:0] mul_full[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];

    // Intermediate wires for adder tree stages
    logic signed [DATA_WIDTH-1:0] add_stage1[4:0];
    logic signed [DATA_WIDTH-1:0] add_stage2[2:0];
    logic signed [DATA_WIDTH-1:0] add_stage3[1:0];
    logic signed [DATA_WIDTH-1:0] final_sum;

    // Perform Multiply-Accumulate
    generate
        for (genvar k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (genvar l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                // Q2.14 multiplication:
                // 1. Multiply two 16-bit signed numbers, resulting in a 32-bit number.
                //    The format of the result is Q4.28.
                assign mul_full[k][l] = image_buffer[k][l] * kernel_matrix[k][l];

                // 2. Arithmetically shift right by 14 bits to convert back to Q2.14 format.
                //    This truncates the lower 14 fractional bits.
                assign mul_results[k][l] = (mul_full[k][l] + ROUND_CONSTANT) >>> 14;
            end
        end
    endgenerate

    // Adder Tree for Q2.14
    // Addition in Q2.14 is standard integer addition.
    // We cast to signed to ensure correct addition.
    assign add_stage1[0] = mul_results[0][0] + mul_results[0][1];
    assign add_stage1[1] = mul_results[0][2] + mul_results[1][0];
    assign add_stage1[2] = mul_results[1][1] + mul_results[1][2];
    assign add_stage1[3] = mul_results[2][0] + mul_results[2][1];
    assign add_stage1[4] = mul_results[2][2];

    assign add_stage2[0] = add_stage1[0] + add_stage1[1];
    assign add_stage2[1] = add_stage1[2] + add_stage1[3];
    assign add_stage2[2] = add_stage1[4];

    assign add_stage3[0] = add_stage2[0] + add_stage2[1];
    assign add_stage3[1] = add_stage2[2];

    assign final_sum = add_stage3[0] + add_stage3[1];

endmodule
