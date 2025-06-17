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
    input logic valid_out,   // Input signal to control when the output is latched

    // Output data
    output logic [DATA_WIDTH-1:0] data_out
);

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
        // If valid_out is low, conv_reg holds its value
    end

    // --- Multiply and Accumulate (MAC) Logic ---

    // Array to hold the 9 intermediate multiplication results
    logic [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE-1:0][KERNEL_SIZE-1:0];

    // Adder tree intermediate summation wires
    logic [DATA_WIDTH-1:0] add_stage1[4:0];
    logic [DATA_WIDTH-1:0] add_stage2[2:0];
    logic [DATA_WIDTH-1:0] add_stage3[1:0];
    logic [DATA_WIDTH-1:0] final_sum;


    // Generate 3x3 = 9 multipliers and the adder tree structure
    genvar k, l;
    generate
        // Instantiate 9 FP16 multipliers
        for (k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                
                // With unpacked 2D arrays, element access is direct and clear.
                // No more complex bit-slicing is needed.
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

        // --- Combinational Adder Tree to Sum the 9 mul_results ---
        // This structure is feed-forward and avoids combinational loops.
        // It sums all 9 products in a single clock cycle.

        // Stage 1: Add pairs of multiplication results
        addfp16 add_s1_0(.a(mul_results[0][0]), .b(mul_results[0][1]), .sum(add_stage1[0]));
        addfp16 add_s1_1(.a(mul_results[0][2]), .b(mul_results[1][0]), .sum(add_stage1[1]));
        addfp16 add_s1_2(.a(mul_results[1][1]), .b(mul_results[1][2]), .sum(add_stage1[2]));
        addfp16 add_s1_3(.a(mul_results[2][0]), .b(mul_results[2][1]), .sum(add_stage1[3]));
        // The 9th result (mul_results[2][2]) passes through this stage
        assign add_stage1[4] = mul_results[2][2];

        // Stage 2: Add results from stage 1
        addfp16 add_s2_0(.a(add_stage1[0]), .b(add_stage1[1]), .sum(add_stage2[0]));
        addfp16 add_s2_1(.a(add_stage1[2]), .b(add_stage1[3]), .sum(add_stage2[1]));
        // The passthrough result from stage 1 passes through again
        assign add_stage2[2] = add_stage1[4];

        // Stage 3: Add results from stage 2
        addfp16 add_s3_0(.a(add_stage2[0]), .b(add_stage2[1]), .sum(add_stage3[0]));
        // Passthrough
        assign add_stage3[1] = add_stage2[2];
        
        // Final Stage: Add the last two results to get the final sum
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
