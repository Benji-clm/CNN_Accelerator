`define CONV_LENGTH 32
`define CONV_OUTPUT 16

module conv_5_hex_test #(
    parameter DATA_WIDTH = 16,    // Width of each data element
    parameter KERNEL_SIZE = 5,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    localparam DATA_ARRAY = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT = 16,   
    parameter CONV_LENGTH = `CONV_LENGTH
)(
    input logic clk,
    input logic rst,
    input logic [DATA_WIDTH-1:0] data_in0,
    input logic [DATA_WIDTH-1:0] data_in1,
    input logic [DATA_WIDTH-1:0] data_in2,
    input logic [DATA_WIDTH-1:0] data_in3,
    input logic [DATA_WIDTH-1:0] data_in4,
    input logic kernel_load,
    input logic valid_in,
    input logic valid_out,
    output logic [DATA_WIDTH-1:0] data_out
);

    // Arrays for kernel and image data
    logic [DATA_ARRAY-1:0] kernel_matrix [0:KERNEL_SIZE-1];
    logic [DATA_ARRAY-1:0] image_buffer [0:KERNEL_SIZE-1];

    // Convolution register
    logic [DATA_WIDTH-1:0] conv_reg;

    // Result register 
    logic [DATA_WIDTH-1:0] result_reg;

    // Assign output
    assign data_out = conv_reg;

    // Load kernel data and process image
    always_ff @(posedge clk) begin
        if (valid_out) begin
            // Output the convolution result
            conv_reg <= result_reg; 
        end else begin 
            // Hold convolution register
            conv_reg <= conv_reg;
        end

        if (rst) begin
            // Reset kernel and image buffers
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                kernel_matrix[i] <= '0;
                image_buffer[i] <= '0;
            end
            // Reset convolution register
            conv_reg <= '0;
        end else if (valid_in) begin
            if (kernel_load) begin
                // Load kernel data into kernel_matrix
                for (int i = 0; i < KERNEL_SIZE-1; i++) begin
                    kernel_matrix[i] <= kernel_matrix[i+1];
                end
                kernel_matrix[KERNEL_SIZE-1] <= {data_in4, data_in3, data_in2, data_in1, data_in0};
            end else begin
                // Move window over the image
                for (int i = 0; i < KERNEL_SIZE-1; i++) begin
                    image_buffer[i] <= image_buffer[i+1];
                end
                image_buffer[KERNEL_SIZE-1] <= {data_in4, data_in3, data_in2, data_in1, data_in0};
            end
        end else begin
            // Hold kernel and image buffers
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                kernel_matrix[i] <= kernel_matrix[i];
                image_buffer[i] <= image_buffer[i];
            end
        end
    end

    logic [DATA_WIDTH-1:0] convolution_result;
    logic [DATA_WIDTH-1:0] multiplication_buffer [0:KERNEL_SIZE*KERNEL_SIZE-1];

    always_comb begin
        // Perform convolution
        // Multiply each pixel in the 5x5 kernel with the corresponding pixel in the image
        multiplication_buffer[0] = image_buffer[0] * kernel_matrix[0];
        multiplication_buffer[1] = image_buffer[1] * kernel_matrix[1];
        multiplication_buffer[2] = image_buffer[2] * kernel_matrix[2];
        multiplication_buffer[3] = image_buffer[3] * kernel_matrix[3];
        multiplication_buffer[4] = image_buffer[4] * kernel_matrix[4];
        multiplication_buffer[5] = image_buffer[5] * kernel_matrix[5];
        multiplication_buffer[6] = image_buffer[6] * kernel_matrix[6];
        multiplication_buffer[7] = image_buffer[7] * kernel_matrix[7];
        multiplication_buffer[8] = image_buffer[8] * kernel_matrix[8];
        multiplication_buffer[9] = image_buffer[9] * kernel_matrix[9];
        multiplication_buffer[10] = image_buffer[10] * kernel_matrix[10];
        multiplication_buffer[11] = image_buffer[11] * kernel_matrix[11];
        multiplication_buffer[12] = image_buffer[12] * kernel_matrix[12];
        multiplication_buffer[13] = image_buffer[13] * kernel_matrix[13];
        multiplication_buffer[14] = image_buffer[14] * kernel_matrix[14];
        multiplication_buffer[15] = image_buffer[15] * kernel_matrix[15];
        multiplication_buffer[16] = image_buffer[16] * kernel_matrix[16];
        multiplication_buffer[17] = image_buffer[17] * kernel_matrix[17];
        multiplication_buffer[18] = image_buffer[18] * kernel_matrix[18];
        multiplication_buffer[19] = image_buffer[19] * kernel_matrix[19];
        multiplication_buffer[20] = image_buffer[20] * kernel_matrix[20];
        multiplication_buffer[21] = image_buffer[21] * kernel_matrix[21];
        multiplication_buffer[22] = image_buffer[22] * kernel_matrix[22];
        multiplication_buffer[23] = image_buffer[23] * kernel_matrix[23];
        multiplication_buffer[24] = image_buffer[24] * kernel_matrix[24];

        // Sum all the products
        convolution_result = multiplication_buffer[0] + multiplication_buffer[1] + multiplication_buffer[2] + multiplication_buffer[3] + multiplication_buffer[4] +
                      multiplication_buffer[5] + multiplication_buffer[6] + multiplication_buffer[7] + multiplication_buffer[8] + multiplication_buffer[9] +
                      multiplication_buffer[10] + multiplication_buffer[11] + multiplication_buffer[12] + multiplication_buffer[13] + multiplication_buffer[14] +
                      multiplication_buffer[15] + multiplication_buffer[16] + multiplication_buffer[17] + multiplication_buffer[18] + multiplication_buffer[19] +
                      multiplication_buffer[20] + multiplication_buffer[21] + multiplication_buffer[22] + multiplication_buffer[23] + multiplication_buffer[24];
    end

    // Calculate convolution using direct hex multiplication and addition
    always_ff @(posedge clk) begin
        if (rst) begin
            result_reg <= '0;
        end else if (valid_in && !kernel_load) begin
            result_reg <= convolution_result;
        end
    end
    

endmodule
