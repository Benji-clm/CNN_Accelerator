module edge_detection_test #(
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
    output logic [DATA_WIDTH-1:0] data_out_0,
    output logic [DATA_WIDTH-1:0] data_out_1,
    output logic [DATA_WIDTH-1:0] data_out_2,
    output logic [DATA_WIDTH-1:0] data_out_3,
    output logic [DATA_WIDTH-1:0] data_out_4,
    output logic [DATA_WIDTH-1:0] data_out_5,
    output logic [DATA_WIDTH-1:0] data_out_6,
    output logic [DATA_WIDTH-1:0] data_out_7,
    output logic [DATA_WIDTH-1:0] data_out_8,
    output logic [DATA_WIDTH-1:0] data_out_9,
    output logic [DATA_WIDTH-1:0] data_out_10,
    output logic [DATA_WIDTH-1:0] data_out_11,
    output logic [DATA_WIDTH-1:0] data_out_12,
    output logic [DATA_WIDTH-1:0] data_out_13,
    output logic [DATA_WIDTH-1:0] data_out_14,
    output logic [DATA_WIDTH-1:0] data_out_15,
    output logic [DATA_WIDTH-1:0] data_out_16,
    output logic [DATA_WIDTH-1:0] data_out_17,
    output logic [DATA_WIDTH-1:0] data_out_18,
    output logic [DATA_WIDTH-1:0] data_out_19,
    output logic [DATA_WIDTH-1:0] data_out_20,
    output logic [DATA_WIDTH-1:0] data_out_21,
    output logic [DATA_WIDTH-1:0] data_out_22,
    output logic [DATA_WIDTH-1:0] data_out_23,
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
    localparam logic signed [DATA_WIDTH-1:0]
        kernel_matrix [0:KERNEL_FLAT_SIZE-1] = '{
            -2, -1,  0,  1,  2,
            -3, -2,  0,  2,  3,
            -4, -3,  0,  3,  4,
            -3, -2,  0,  2,  3,
            -2, -1,  0,  1,  2
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
    logic signed [DATA_WIDTH-1:0] data_out [0:IMAGE_SIZE-KERNEL_SIZE];

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

    assign data_out_0 = data_out[0];
    assign data_out_1 = data_out[1];
    assign data_out_2 = data_out[2];
    assign data_out_3 = data_out[3];
    assign data_out_4 = data_out[4];
    assign data_out_5 = data_out[5];
    assign data_out_6 = data_out[6];
    assign data_out_7 = data_out[7];
    assign data_out_8 = data_out[8];
    assign data_out_9 = data_out[9];
    assign data_out_10 = data_out[10];
    assign data_out_11 = data_out[11];
    assign data_out_12 = data_out[12];
    assign data_out_13 = data_out[13];
    assign data_out_14 = data_out[14];
    assign data_out_15 = data_out[15];
    assign data_out_16 = data_out[16];
    assign data_out_17 = data_out[17];
    assign data_out_18 = data_out[18];
    assign data_out_19 = data_out[19];
    assign data_out_20 = data_out[20];
    assign data_out_21 = data_out[21];
    assign data_out_22 = data_out[22];
    assign data_out_23 = data_out[23];

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
