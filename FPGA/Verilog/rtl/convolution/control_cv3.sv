`define CONV_LENGTH 32
`define CONV_OUTPUT 32

module control_cv3 #(
    parameter DATA_WIDTH = 16,
    parameter KERNEL_SIZE = 3,
    parameter IMAGE_SIZE = 12,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    localparam DATA_ARRAY = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT = `CONV_OUTPUT,
    parameter CONV_LENGTH = `CONV_LENGTH
)(
    input logic clk,
    input logic rst,
    input logic start,
    output logic [CONV_OUTPUT-1:0] data_out [IMAGE_SIZE-KERNEL_SIZE:0],
    output logic done
);

// Define states for kernel loading and image processing
typedef enum logic [2:0] {
    IDLE,
    LOAD_KERNEL,
    KERNEL_DELAY,
    PROCESS_IMAGE,
    COMPLETE
} state_t;

state_t current_state, next_state;

logic signed [DATA_ARRAY-1:0] kernel_matrix [0:KERNEL_SIZE-1][0:KERNEL_SIZE-1];
logic signed [DATA_ARRAY-1:0] image [0:IMAGE_SIZE-1][0:IMAGE_SIZE-1];

// control logic for convolution operation parallelization
logic kernel_load;
logic valid_in;
logic valid_out;

//column counters for kernel and image loading and processing
logic [$clog2(KERNEL_SIZE):0] kernel_col;
logic [$clog2(IMAGE_SIZE):0] image_col;
logic [$clog2(KERNEL_SIZE):0] delay_counter;  // For the image loading delay

initial begin
    // Initialize kernel matrix and image buffer (example values)
    $readmemh("kernel_data.hex", kernel_matrix);
    $readmemh("image_data.hex", image);
    // Do not assign kernel_load, valid_in, valid_out here for synthesis
end

// State register
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        current_state <= IDLE;
        kernel_col <= 0;
        image_col <= 0;
        delay_counter <= 0;
        kernel_load <= 0;
        valid_in <= 0;
        valid_out <= 0;
        done <= 0;
    end else begin
        current_state <= next_state;
        
        case (current_state)
            IDLE: begin
                // Reset all counters and signals
                kernel_col <= 0;
                image_col <= 0;
                delay_counter <= 0;
                kernel_load <= 0;
                valid_in <= 0;
                valid_out <= 0;
                done <= 0;
            end
            
            LOAD_KERNEL: begin
                // Set kernel_load high during kernel loading
                kernel_load <= 1;
                valid_in <= 1;
                valid_out <= 0; // Set valid output low during kernel loading
                // Update kernel position counters
               if (kernel_col < KERNEL_SIZE-1) begin
                    kernel_col <= kernel_col + 1;
                end 
            end
            
            KERNEL_DELAY: begin
                // Maintain kernel_load high during delay
                kernel_load <= 0;
                valid_in <= 1;
                image_col <= image_col + 1; // Increment image column counter
                // Count delay cycles
                delay_counter <= delay_counter + 1;
                valid_out <= 0; // Set valid output low during delay
            end
            
            PROCESS_IMAGE: begin
                // Switch to processing image data
                kernel_load <= 0;
                valid_in <= 1;
                valid_out <= 1;  // Valid output after the delay
                
                // Slide the kernel across the image
                if (image_col < IMAGE_SIZE - KERNEL_SIZE + 1) begin
                    image_col <= image_col + STRIDE;
                end 
            end
            COMPLETE: begin
                done <= 1;
                valid_out <= 0;
            end
        endcase
    end
end

always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        image_col <= 0; // Reset valid output
        kernel_load <= 1'b1; // Reset kernel load signal
    end else begin
        if (kernel_load) begin
            //load kernel data
        end
        else if (image_col < IMAGE_SIZE - 1) begin
            // Increment column counter to slide the kernel
            image_col <= image_col + STRIDE;
        end else begin
            image_col <= 0; // Reset counter after processing all rows
        end
    end
end

// Next state logic
always_comb begin
    next_state = current_state;
    
    case (current_state)
        IDLE: begin
            if (start)
                next_state = LOAD_KERNEL;
        end
        
        LOAD_KERNEL: begin
            if (kernel_col == KERNEL_SIZE-1)
                next_state = KERNEL_DELAY; // Reset image column counter
        end
        
        KERNEL_DELAY: begin
            if (delay_counter >= KERNEL_SIZE - 3)  // 2-cycle delay
                next_state = PROCESS_IMAGE;
        end
        
        PROCESS_IMAGE: begin
            if (image_col == IMAGE_SIZE - 1)
                next_state = COMPLETE;
        end
        
        COMPLETE: begin
            if (!start)
                next_state = IDLE;
        end
    endcase
end

// data multiplexing for kernel and image data

logic [DATA_ARRAY-1:0] data_in [IMAGE_SIZE-1:0];

always_comb begin
    if (kernel_load) begin
        for (int i = 0; i < KERNEL_SIZE; i++) begin
            // Load kernel data into data_in0, data_in1, data_in2, data_in3, data_in4
            data_in[i] = kernel_matrix[kernel_col][i];
        end
    end else begin
        for (int i = 0; i < IMAGE_SIZE; i++) begin
            data_in[i] = image[image_col][i]; 
        end
    end
end

// create a parallel block for each row of the image
genvar i;
generate
    for (i = 0; i < IMAGE_SIZE - KERNEL_SIZE + 1; i++) begin : gen_parallel
        conv_3 #(
            .DATA_WIDTH(DATA_WIDTH),
            .KERNEL_SIZE(KERNEL_SIZE),
            .STRIDE(STRIDE),
            .PADDING(PADDING)
        ) conv_inst (
            .clk(clk),
            .rst(rst),
            .data_in0(data_in[(kernel_load ? 0 : i * STRIDE) + 0]),
            .data_in1(data_in[(kernel_load ? 0 : i * STRIDE) + 1]),
            .data_in2(data_in[(kernel_load ? 0 : i * STRIDE) + 2]),
            .kernel_load(kernel_load),
            .valid_in(valid_in),
            .valid_out(valid_out),
            .data_out(data_out[i])
        );
    end
endgenerate

endmodule
