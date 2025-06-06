`define CONV_LENGTH 32
`define CONV_OUTPUT 16

module convolution #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    localparam DATA_ARRAY = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT = `CONV_OUTPUT,
    parameter CONV_LENGTH = `CONV_LENGTH
)(
    input logic clk,
    input logic rst,
    input logic [DATA_WIDTH-1:0] data_in0,
    input logic [DATA_WIDTH-1:0] data_in1,
    input logic [DATA_WIDTH-1:0] data_in2,
    input logic kernel_load,
    input logic valid_in,
    input logic valid_out,
    output logic [CONV_OUTPUT-1:0] data_out
);

    // arrays for kernel and image data
    logic signed [DATA_ARRAY-1:0] kernel_matrix [0:KERNEL_SIZE-1];
    logic signed [DATA_ARRAY-1:0] image_buffer [0:KERNEL_SIZE-1];

    // convolution register - now 16 bits for FP16
    logic signed [DATA_WIDTH-1:0] conv_reg;

    // result register - now 16 bits for FP16
    logic signed [DATA_WIDTH-1:0] result_reg;

    // Assign output
    assign data_out = conv_reg;

    // Load kernel data
    always_ff @(posedge clk) begin
        if (valid_out) begin
            // output the convolution result
            conv_reg <= result_reg;
        end else begin 
            // hold convolution register
            conv_reg <= conv_reg;
        end

        if (rst) begin
            //reset kernel and image buffers
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                kernel_matrix[i] <= '0;
                image_buffer[i] <= '0;
            end
            //reset convolution register
            conv_reg <= '0;
        end else if (valid_in) begin
            if (kernel_load) begin
                // Load kernel data into kernel_matrix
                for (int i = 0; i < KERNEL_SIZE-1; i++) begin
                    kernel_matrix[i] <= kernel_matrix[i+1];
                end
                kernel_matrix[KERNEL_SIZE-1] <= {data_in2, data_in1, data_in0};
            end else begin
                // move window over the image
                for (int i = 0; i < KERNEL_SIZE-1; i++) begin
                    image_buffer[i] <= image_buffer[i+1];
                end
                image_buffer[KERNEL_SIZE-1] <= {data_in2, data_in1, data_in0};
            end
        end else begin
            // hold kernel and image buffers
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                kernel_matrix[i] <= kernel_matrix[i];
                image_buffer[i] <= image_buffer[i];
            end
        end
    end

    // First, declare all multiplier results outside the generate block
    // All using DATA_WIDTH (16 bits) for FP16
    logic signed [DATA_WIDTH-1:0] mul_results[KERNEL_SIZE][KERNEL_SIZE];
    logic signed [DATA_WIDTH-1:0] partial_sums[KERNEL_SIZE][KERNEL_SIZE];

    // Initialize result_reg at the beginning of each operation
    always_ff @(posedge clk) begin
        if (rst) begin
            result_reg <= '0;
        end else if (!kernel_load) begin
            // Only update when processing image, not when loading kernel
            result_reg <= partial_sums[KERNEL_SIZE-1][KERNEL_SIZE-1];
        end
    end

    // Generate multiplier and adder instances
    genvar k, l;
    generate
        for (k = 0; k < KERNEL_SIZE; k++) begin : gen_row
            for (l = 0; l < KERNEL_SIZE; l++) begin : gen_col
                // Extract specific elements from the arrays
                wire [DATA_WIDTH-1:0] img_element = image_buffer[k][(l+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                wire [DATA_WIDTH-1:0] kernel_element = kernel_matrix[k][(l+1)*DATA_WIDTH-1 -: DATA_WIDTH];
                
                // Instantiate multiplier
                mulfp16 mul_inst (
                    .a_in(img_element),     // Match port names to your mulfp16 module
                    .b_in(kernel_element),  
                    .c_out(mul_results[k][l])  // Store in the 2D array
                );
                
                // Create a tree of adders 
                // First element initialization
                if (k == 0 && l == 0) begin
                    assign partial_sums[0][0] = mul_results[0][0];
                end
                // First row additions
                else if (k == 0 && l > 0) begin
                    addfp16 add_inst (
                        .a(partial_sums[0][l-1]),
                        .b(mul_results[0][l]),
                        .sum(partial_sums[0][l])
                    );
                end
                // First column additions
                else if (l == 0 && k > 0) begin
                    addfp16 add_inst (
                        .a(partial_sums[k-1][KERNEL_SIZE-1]),
                        .b(mul_results[k][0]),
                        .sum(partial_sums[k][0])
                    );
                end
                // All other positions
                else begin
                    addfp16 add_inst (
                        .a(partial_sums[k][l-1]),
                        .b(mul_results[k][l]),
                        .sum(partial_sums[k][l])
                    );
                end
            end
        end
    endgenerate

endmodule