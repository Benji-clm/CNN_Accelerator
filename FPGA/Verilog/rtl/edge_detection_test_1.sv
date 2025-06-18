module edge_detection_test_1 #(
    parameter DATA_WIDTH = 16,    // Half-precision float width
    parameter KERNEL_SIZE = 5,
    parameter STRIDE = 1,
    parameter PADDING = 1,
    localparam DATA_ARRAY = DATA_WIDTH * KERNEL_SIZE,
    parameter CONV_OUTPUT = 16,   // Changed to match DATA_WIDTH for FP16
    parameter IMAGE_SIZE = 28
)(
    input logic clk,
    input logic rst,
    input logic start,
    input logic [255:0] data_in,
    output logic write_enable,
    output logic read_enable,
    output logic [DATA_WIDTH-1:0] data_out_x [IMAGE_SIZE-KERNEL_SIZE:0],
    output logic [11:0] addr,
    output logic done,
    output logic valid_out_col,
    output logic [$clog2(IMAGE_SIZE):0] out_col_num,
    output logic keep
);

    //================================================================
    // 1. PARAMETERS and KERNEL DEFINITION
    //================================================================
    localparam KERNEL_FLAT_SIZE = KERNEL_SIZE * KERNEL_SIZE;
    localparam logic [DATA_WIDTH-1:0]
        kernel_matrix [0:KERNEL_FLAT_SIZE-1] = '{
            16'h31E5, 16'h3542, 16'h36B2, 16'h31F9, 16'hAE40, 
            16'hB71D, 16'hB5E7, 16'hB4E5, 16'hACCC, 16'h2B28, 
            16'hB11A, 16'hB611, 16'hB541, 16'hA9FB, 16'h2D56, 
            16'hB19F, 16'hB25A, 16'h3661, 16'h2C89, 16'hB1C0, 
            16'h38F3, 16'h38B0, 16'h2E98, 16'hB041, 16'h33B7
        };

    //================================================================
    // 2. STATE MACHINE and INTERNAL SIGNALS
    //================================================================
    typedef enum logic [1:0] { IDLE, LOAD_KERNEL, PROCESS_IMAGE, COMPLETE } state_t;

    state_t current_state, next_state;
    logic kernel_load;
    logic valid_in;
    logic valid_out;
    logic [$clog2(IMAGE_SIZE):0] image_col;
    logic [$clog2(IMAGE_SIZE):0] image_row;
    logic [$clog2(KERNEL_SIZE):0] kernel_col;
    logic [$clog2(KERNEL_SIZE):0] kernel_row;
    logic [DATA_WIDTH-1:0] data_buffer [0:IMAGE_SIZE-1];
    logic signed [DATA_WIDTH-1:0] data_out [IMAGE_SIZE-KERNEL_SIZE:0];

    //================================================================
    // 3. COMBINATIONAL LOGIC (Single always_comb block)
    //================================================================
    always_comb begin
        // Default assignments to prevent latches
        next_state = current_state;
        kernel_load = 1'b0;
        valid_out = 1'b0;
        read_enable = 1'b0;
        
        // State transition logic
        case (current_state)
            IDLE: begin
                if (start) next_state = LOAD_KERNEL;
            end
            
            LOAD_KERNEL: begin
                kernel_load = 1'b1;
                if (kernel_col == KERNEL_SIZE -1 && kernel_row == KERNEL_SIZE - 1) begin
                    next_state = PROCESS_IMAGE;
                end
            end
            
            PROCESS_IMAGE: begin
                read_enable = 1'b1;
                if (image_col >= KERNEL_SIZE) valid_out = 1'b1; 
                if (image_col == IMAGE_SIZE - 1 && image_row >= IMAGE_SIZE - 16) begin
                    next_state = COMPLETE;
                end
            end
            
            COMPLETE: begin
                valid_out = 1'b1;
                next_state = IDLE;
            end
        endcase

        // Other combinational assignments
        addr = (image_col * 2 + (image_row >= 16));
        valid_out_col = valid_in && valid_out;
        out_col_num = image_col - KERNEL_SIZE + 1;
    end

    //================================================================
    // 4. SEQUENTIAL LOGIC (Single always_ff block)
    //================================================================
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= IDLE;
            image_col <= '0;
            image_row <= '0;
            kernel_col <= '0;
            kernel_row <= '0;
            done <= 1'b0;
        end else begin
            // Default assignments for registers
            done <= 1'b0;
            valid_in <= 1'b0;

            // State register update
            current_state <= next_state;

            // Action logic based on CURRENT state
            case (current_state)
                IDLE: begin
                    if (start) begin
                        kernel_col <= '0;
                        kernel_row <= '0;
                        image_row <= '0;
                        image_col <= '0;
                    end
                end

                LOAD_KERNEL: begin
                    logic [$clog2(KERNEL_SIZE):0] next_kernel_row;
                    logic [$clog2(KERNEL_SIZE):0] next_kernel_col;

                    if (kernel_row < KERNEL_SIZE - 1) begin
                        next_kernel_row = kernel_row + 1;
                        next_kernel_col = kernel_col;
                    end else begin
                        next_kernel_row = 0;
                        next_kernel_col = kernel_col + 1;
                        valid_in <= 1'b1;
                    end
                    kernel_row <= next_kernel_row;
                    kernel_col <= next_kernel_col;
                    data_buffer[next_kernel_row] <= kernel_matrix[next_kernel_row * KERNEL_SIZE + next_kernel_col];
                end

                PROCESS_IMAGE: begin
                    for (int j = 0; j < 16; j++) begin
                        if (image_row + j < IMAGE_SIZE) begin
                            data_buffer[image_row + j] <= data_in[j * DATA_WIDTH +: DATA_WIDTH];
                        end
                    end
                    
                    if (image_row < IMAGE_SIZE - 16) begin
                        image_row <= image_row + 16;
                    end else if (image_col < IMAGE_SIZE) begin // FIXED off-by-one error
                        image_row <= '0;
                        image_col <= image_col + 1;
                        valid_in <= 1'b1;
                    end 
                end

                COMPLETE: begin
                    done <= 1'b1;
                end
            endcase
        end
    end

    //================================================================
    // 5. OUTPUT ASSIGNMENTS and INSTANTIATIONS
    //================================================================
    assign write_enable = 1;
    assign data_out_x = data_out;

    generate
        for (genvar i = 0; i < IMAGE_SIZE - KERNEL_SIZE + 1; i++) begin : gen_parallel
            conv_5_hex_test #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE),
                .STRIDE(STRIDE)
            ) conv_inst (
                .clk(clk),
                .rst(rst),
                .data_in0(data_buffer[(kernel_load ? 0 : i * STRIDE) + 0]),
                .data_in1(data_buffer[(kernel_load ? 0 : i * STRIDE) + 1]),
                .data_in2(data_buffer[(kernel_load ? 0 : i * STRIDE) + 2]),
                .data_in3(data_buffer[(kernel_load ? 0 : i * STRIDE) + 3]),
                .data_in4(data_buffer[(kernel_load ? 0 : i * STRIDE) + 4]),
                .kernel_load(kernel_load),
                .valid_in(valid_in),
                .valid_out(valid_out),
                .data_out(data_out[i])
            );
        end
    endgenerate

    logic ka_r;
    always_ff @(posedge clk or posedge rst) begin
        if (rst)  ka_r <= 1'b0;
        else      ka_r <= ~ka_r;
    end
    assign keep = ka_r;

endmodule
