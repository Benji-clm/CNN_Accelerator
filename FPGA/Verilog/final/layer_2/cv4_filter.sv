module cv4_filter #(
    parameter DATA_WIDTH      = 16, // Data width for each element (FP16)
    parameter KERNEL_SIZE     = 4,  // The size of the convolution kernel (e.g., 4 for 4x4)
    parameter INPUT_COL_SIZE  = 5, // Height of the input data column
    localparam PARALLEL_UNITS = INPUT_COL_SIZE - KERNEL_SIZE + 1
)(
    input logic clk,
    input logic rst,

    // --- Control Signals ---
    input logic kernel_load,    // Assert to load kernel weights, de-assert for image processing
    input logic valid_in,       // Assert when input_column and kernel_column are valid

    // --- Data Inputs ---
    input logic [DATA_WIDTH-1:0] input_column [INPUT_COL_SIZE-1:0],
    input logic [DATA_WIDTH-1:0] kernel_column [KERNEL_SIZE-1:0],

    // --- Data Outputs ---
    output logic [DATA_WIDTH-1:0] output_column [PARALLEL_UNITS-1:0],
    output logic valid_out
);

    // --- Fully Synchronous Pipeline Control Logic ---

    // State machine to track when the pipeline is primed with enough data.
    // For a KERNEL_SIZE of 4, we need 3 valid columns before we can compute.
    typedef enum logic [1:0] {
        S_IDLE,    // 0 valid columns in the pipe.
        S_HAVE_1,  // 1 valid column in the pipe.
        S_HAVE_2,  // 2 valid columns in the pipe.
        S_HAVE_3   // 3+ valid columns in the pipe, ready for computation.
    } state_t;

    state_t current_state, next_state;
    logic   computation_trigger;
    logic   trigger_d1, trigger_d2;

    // State Register (Sequential, Posedge only, Asynchronous Reset)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= S_IDLE;
        end else if (valid_in && !kernel_load) begin
            current_state <= next_state;
        end
    end

    // Next State Logic (Combinational)
    always_comb begin
        case (current_state)
            S_IDLE:   next_state = S_HAVE_1;
            S_HAVE_1: next_state = S_HAVE_2;
            S_HAVE_2: next_state = S_HAVE_3;
            default:  next_state = S_HAVE_3; // Saturate here once pipeline is full
        endcase
    end

    // A computation is triggered for one clock cycle when the pipeline is full
    // and the fourth valid input column arrives.
    assign computation_trigger = (current_state == S_HAVE_3) && valid_in && !kernel_load;

    // Create a 2-cycle delay pipeline for the valid signal. This delay aligns the
    // final `valid_out` signal with the data path latency of the conv_4 modules.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            trigger_d1 <= 1'b0;
            trigger_d2 <= 1'b0;
        end else begin
            trigger_d1 <= computation_trigger;
            trigger_d2 <= trigger_d1;
        end
    end

    // The layer's final output valid signal is the fully pipelined trigger.
    assign valid_out = trigger_d2;


    // --- Parallel Convolution Units ---
    genvar i;
    generate
        for (i = 0; i < PARALLEL_UNITS; i++) begin : gen_conv_units
            
            wire [DATA_WIDTH-1:0] conv_data_slice [KERNEL_SIZE-1:0];
            assign conv_data_slice[0] = input_column[i];
            assign conv_data_slice[1] = input_column[i+1];
            assign conv_data_slice[2] = input_column[i+2];
            assign conv_data_slice[3] = input_column[i+3];
            wire [DATA_WIDTH-1:0] muxed_data_in [KERNEL_SIZE-1:0];
        
            genvar j;
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin : gen_mux
                assign muxed_data_in[j] = kernel_load ? kernel_column[j] : conv_data_slice[j];
            end

            // Instantiate the 4x4 convolution filter
            conv_4 #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE)
            ) conv_inst (
                .clk(clk),
                .rst(rst),
                .data_in(muxed_data_in),
                .kernel_load(kernel_load),
                .valid_in(valid_in),
                // Connect the correctly timed, single-edge, synchronous valid signal.
                // This port is likely an input, misnamed in the original code.
                .valid_out(trigger_d1),
                .data_out(output_column[i])
            );
        end
    endgenerate

endmodule