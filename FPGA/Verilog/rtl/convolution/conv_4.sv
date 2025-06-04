`define CONV_LENGTH 32
`define CONV_OUTPUT 32

module convolution #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 4,
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
    input logic [DATA_WIDTH-1:0] data_in3,
    input logic kernel_load,
    input logic valid_in,
    input logic valid_out,
    output logic [CONV_OUTPUT-1:0] data_out
);

    // arrays for kernel and image data
    logic signed [DATA_ARRAY-1:0] kernel_matrix [0:KERNEL_SIZE-1];
    logic signed [DATA_ARRAY-1:0] image_buffer [0:KERNEL_SIZE-1];

    // convolution register
    logic signed [CONV_OUTPUT-1:0] conv_reg;

    // result register
    logic signed [CONV_OUTPUT-1:0] result_reg;

    // Multiplication store array
    logic signed [DATA_ARRAY-1:0] mul_store [0:KERNEL_SIZE*KERNEL_SIZE-1];

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
                for (int i = 0; i < KERNEL_SIZE; i++) begin
                    kernel_matrix[i] <= kernel_matrix[i+1];
                end
                kernel_matrix[KERNEL_SIZE-1] <= {data_in3, data_in2, data_in1, data_in0};
            end else begin
                // move window over the image
                for (int i = 0; i < KERNEL_SIZE-1; i++) begin
                    image_buffer[i] <= image_buffer[i+1];
                end
                image_buffer[KERNEL_SIZE-1] <= {data_in3, data_in2, data_in1, data_in0};
            end
        end else begin
            // hold kernel and image buffers
            for (int i = 0; i < KERNEL_SIZE; i++) begin
                kernel_matrix[i] <= kernel_matrix[i];
                image_buffer[i] <= image_buffer[i];
            end
        end
    end

    
    // This section is commented out as it is replaced by the fp16 multiplier instances
    // Convolution operation
    always_comb begin
        result_reg = '0;
        for (int i = 0; i < KERNEL_SIZE; i++) begin
            for (int j = 0; j < KERNEL_SIZE; j++) begin
                result_reg += $signed(kernel_matrix[i][(j+1)*DATA_WIDTH-1 -: DATA_WIDTH]) * 
                              $signed(image_buffer[i][(j+1)*DATA_WIDTH-1 -: DATA_WIDTH]);
            end
        end
    end 
    

/*
genvar k, l;
// Generate fp16 multiplier instances
generate
    for (k = 0; k < KERNEL_SIZE; k++) begin : gen_row
        for (l = 0; l < KERNEL_SIZE; l++) begin : gen_col
            // Generate multiplier instances for each kernel and image buffer
            // This assumes that the kernel and image buffers are 2D arrays
            // and that the multiplication is done in parallel.
            mulfp16 #(
            ) mul_inst (
                .clk(clk),
                .rst(rst),
                .a_in(image_buffer[k]),
                .b_in(kernel_matrix[l]),
                .load_b(0),
                .a_out(),
                .c_out(mul_store[k*KERNEL_SIZE + l]),
                .mode_fp16(1'b1), // Assuming fp16 mode
                .signed_mode(1'b1) // Assuming signed mode
            );
        end
    end
endgenerate
*/


    endmodule
