module conv_4 #(
    // Parameters for configuring the convolution operation
    parameter DATA_WIDTH = 16, // Data width for each pixel/weight (e.g., 16 for FP16)
    parameter KERNEL_SIZE = 4   // Size of the convolution kernel (4 for a 4x4 kernel)
)(
    input logic clk,
    input logic rst,

    // Input data streams (representing one row of the image/kernel)
    input logic [DATA_WIDTH-1:0] data_in [KERNEL_SIZE-1:0],

    // Control signals
    input logic kernel_load, // High to load kernel weights, low to process image data
    input logic valid_in,    // High when input data is valid
    input logic valid_out,   // Input signal to control when the output is latched

    // Output data
    output logic [DATA_WIDTH-1:0] data_out
);

    // Unpacked 4x4 2D Arrays for Kernel and Image Window
    logic [DATA_WIDTH-1:0] kernel_matrix [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];
    logic [DATA_WIDTH-1:0] image_buffer  [KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];

    // Output register that is updated based on the external valid_out signal
    logic [DATA_WIDTH-1:0] conv_reg;
    // Register to hold the final result of the convolution from the adder tree
    logic [DATA_WIDTH-1:0] result_reg;

    // Combinational assignment of the output register to the port
    assign data_out = conv_reg;

    // This block handles shifting data into the kernel and image buffers
    always_ff @(posedge clk) begin
        // Reset has the highest priority
        if (rst) begin
            // Reset kernel and image buffers to all zeros
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    kernel_matrix[i][j] <= '0;
                    image_buffer[i][j]  <= '0;
                end
            end
            // Reset convolution output register
            conv_reg <= '0;
        end 
        // Only process data if the input is valid
        else if (valid_in) begin
            if (kernel_load) begin
                // Load kernel data: shift rows up and load new row at the bottom
                for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                    kernel_matrix[i] <= kernel_matrix[i+1]; // Assign whole row
                end
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    kernel_matrix[KERNEL_SIZE-1][j] <= data_in[j];
                end
            end else begin
                // Load image data: shift rows up and load new row at the bottom
                for (int i = 0; i < KERNEL_SIZE - 1; i++) begin
                    image_buffer[i] <= image_buffer[i+1]; // Assign whole row
                end
                for (int j = 0; j < KERNEL_SIZE; j++) begin
                    image_buffer[KERNEL_SIZE-1][j] <= data_in[j];
                end
            end
        end
    end
    
    // This block updates the final output register (conv_reg) based on valid_out
    always_ff @(posedge clk) begin
        if (rst) begin
             conv_reg <= '0;
        end else if (valid_out) begin
            // Output the convolution result
            conv_reg <= result_reg;
        end
    end

    // --- Multiply and Accumulate (MAC) Logic ---

    // Array to hold the 16 intermediate multiplication results
    logic [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];

    // Adder tree intermediate summation wires
    logic [DATA_WIDTH-1:0] add_stage1[7:0];
    logic [DATA_WIDTH-1:0] add_stage2[3:0];
    logic [DATA_WIDTH-1:0] add_stage3[1:0];
    logic [DATA_WIDTH-1:0] final_sum;


    // Generate 4x4 = 16 multipliers and the adder tree structure
    genvar k, l;
    generate
        // Instantiate 16 FP16 multipliers
        for (k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                
                // Direct 2D array element access
                wire [DATA_WIDTH-1:0] img_element    = image_buffer[k][l];
                wire [DATA_WIDTH-1:0] kernel_element = kernel_matrix[k][l];
                
                // Instantiate multiplier (ensure port names match your mulfp16 module)
                mulfp16 mul_inst (
                    .a_in(img_element),
                    .b_in(kernel_element),
                    .c_out(mul_results[k][l])
                );
            end
        end

        // --- Combinational Adder Tree to Sum the 16 mul_results ---
        // This balanced tree structure sums all 16 products in a single clock cycle.

        // Stage 1: 16 inputs -> 8 outputs (8 adders)
        addfp16 add_s1_0(.a(mul_results[0][0]), .b(mul_results[0][1]), .sum(add_stage1[0]));
        addfp16 add_s1_1(.a(mul_results[0][2]), .b(mul_results[0][3]), .sum(add_stage1[1]));
        addfp16 add_s1_2(.a(mul_results[1][0]), .b(mul_results[1][1]), .sum(add_stage1[2]));
        addfp16 add_s1_3(.a(mul_results[1][2]), .b(mul_results[1][3]), .sum(add_stage1[3]));
        addfp16 add_s1_4(.a(mul_results[2][0]), .b(mul_results[2][1]), .sum(add_stage1[4]));
        addfp16 add_s1_5(.a(mul_results[2][2]), .b(mul_results[2][3]), .sum(add_stage1[5]));
        addfp16 add_s1_6(.a(mul_results[3][0]), .b(mul_results[3][1]), .sum(add_stage1[6]));
        addfp16 add_s1_7(.a(mul_results[3][2]), .b(mul_results[3][3]), .sum(add_stage1[7]));

        // Stage 2: 8 inputs -> 4 outputs (4 adders)
        addfp16 add_s2_0(.a(add_stage1[0]), .b(add_stage1[1]), .sum(add_stage2[0]));
        addfp16 add_s2_1(.a(add_stage1[2]), .b(add_stage1[3]), .sum(add_stage2[1]));
        addfp16 add_s2_2(.a(add_stage1[4]), .b(add_stage1[5]), .sum(add_stage2[2]));
        addfp16 add_s2_3(.a(add_stage1[6]), .b(add_stage1[7]), .sum(add_stage2[3]));

        // Stage 3: 4 inputs -> 2 outputs (2 adders)
        addfp16 add_s3_0(.a(add_stage2[0]), .b(add_stage2[1]), .sum(add_stage3[0]));
        addfp16 add_s3_1(.a(add_stage2[2]), .b(add_stage2[3]), .sum(add_stage3[1]));

        // Final Stage: 2 inputs -> 1 output (1 adder)
        addfp16 add_final(.a(add_stage3[0]), .b(add_stage3[1]), .sum(final_sum));

    endgenerate

    // Register the final combinational sum
    always_ff @(posedge clk) begin
        if (rst) begin
            result_reg <= '0;
        end else if (!kernel_load) begin
            // Only update the result when processing image data
            result_reg <= final_sum;
        end
    end

endmodule
