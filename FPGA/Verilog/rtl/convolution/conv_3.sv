module conv_3 #(
    // Parameters for configuring the convolution operation
    parameter DATA_WIDTH = 16, // Data width for each pixel/weight (e.g., 16 for FP16)
    parameter KERNEL_SIZE = 3 // Size of the convolution kernel (3 for a 3x3 kernel)
)(
    input logic clk,
    input logic rst,

    // Input data streams (representing one column of the image)
    input logic [DATA_WIDTH-1:0] data_in [KERNEL_SIZE - 1: 0],

    // Control signals
    input logic kernel_load, // High to load kernel weights, low to process image data
    input logic valid_in,    // High when input data is valid
    output logic valid_out,   // FIXED: Port direction is now output
    // Input signal to control when the output is latched

    // Output data
    output logic [DATA_WIDTH-1:0] data_out
);

    // Internal Registers
    logic [DATA_WIDTH-1:0] kernel_matrix [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] image_buffer  [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] conv_reg;
    logic [DATA_WIDTH-1:0] result_reg;
    logic                  valid_s1, valid_s2; // FIXED: Internal pipeline for valid signal

    // Combinational assignment of the output register to the port
    assign data_out = conv_reg;
    // FIXED: Assign the final stage of the valid pipeline to the output port
    assign valid_out = valid_s2;

    always_ff @(posedge clk or posedge rst) begin
        // Reset has the highest priority and is asynchronous
        if (rst) begin
            // Reset kernel and image buffers to all zeros
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    kernel_matrix[i][j] <= '0;
                    image_buffer[i][j]  <= '0;
                end
            end
            conv_reg   <= '0;
            result_reg <= '0;
            // FIXED: Reset the valid pipeline registers
            valid_s1 <= 1'b0;
            valid_s2 <= 1'b0;
        end
        // All clocked logic is in the else block
        else begin
            // Input data shifting logic
            if (valid_in) begin
                if (kernel_load) begin
                    // Load kernel data: shift rows up and load new row at the bottom
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        kernel_matrix[i] <= kernel_matrix[i+1];
                    end
                    for (int j = 0; j < KERNEL_SIZE; j++) begin
                        kernel_matrix[KERNEL_SIZE-1][j] <= data_in[j];
                    end
                end else begin
                    // Load image data: shift rows up and load new row at the bottom
                    for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                        image_buffer[i] <= image_buffer[i+1];
                    end
                    for (int j = 0; j < KERNEL_SIZE; j++) begin
                        image_buffer[KERNEL_SIZE-1][j] <= data_in[j];
                    end
                end
            end

            // FIXED: A 2-stage valid pipeline to match the 2-stage data pipeline (result_reg -> conv_reg)
            valid_s1 <= valid_in && !kernel_load;
            valid_s2 <= valid_s1;

            // Latch the first stage result when its corresponding valid is high
            if (valid_s1) begin
                result_reg <= final_sum;
            end

            // Latch the final output when its corresponding valid is high
            if (valid_s2) begin
                conv_reg <= result_reg;
            end
        end
    end

    // --- Multiply and Accumulate (MAC) Logic ---
    // This part is purely combinational and remains unchanged.

    logic [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] add_stage1[4:0];
    logic [DATA_WIDTH-1:0] add_stage2[2:0];
    logic [DATA_WIDTH-1:0] add_stage3[1:0];
    logic [DATA_WIDTH-1:0] final_sum;

    genvar k, l;
    generate
        for (k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                mulfp16 mul_inst (
                    .a_in(image_buffer[k][l]),
                    .b_in(kernel_matrix[k][l]),
                    .c_out(mul_results[k][l])
                );
            end
        end

        // Combinational Adder Tree
        addfp16 add_s1_0(.a(mul_results[0][0]), .b(mul_results[0][1]), .sum(add_stage1[0]));
        addfp16 add_s1_1(.a(mul_results[0][2]), .b(mul_results[1][0]), .sum(add_stage1[1]));
        addfp16 add_s1_2(.a(mul_results[1][1]), .b(mul_results[1][2]), .sum(add_stage1[2]));
        addfp16 add_s1_3(.a(mul_results[2][0]), .b(mul_results[2][1]), .sum(add_stage1[3]));
        assign add_stage1[4] = mul_results[2][2];

        addfp16 add_s2_0(.a(add_stage1[0]), .b(add_stage1[1]), .sum(add_stage2[0]));
        addfp16 add_s2_1(.a(add_stage1[2]), .b(add_stage1[3]), .sum(add_stage2[1]));
        assign add_stage2[2] = add_stage1[4];

        addfp16 add_s3_0(.a(add_stage2[0]), .b(add_stage2[1]), .sum(add_stage3[0]));
        assign add_stage3[1] = add_stage2[2];
        
        addfp16 add_final(.a(add_stage3[0]), .b(add_stage3[1]), .sum(final_sum));
    endgenerate

endmodule
