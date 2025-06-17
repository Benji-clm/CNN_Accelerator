module cv4_filter #(
    parameter DATA_WIDTH      = 16, // Data width for each element (FP16)
    parameter KERNEL_SIZE     = 4,  // The size of the convolution kernel (e.g., 3 for 3x3)
    parameter INPUT_COL_SIZE  = 5, // Height of the input data column
    localparam PARALLEL_UNITS = INPUT_COL_SIZE - KERNEL_SIZE + 1
)(
    input logic clk,
    input logic rst,

    // --- Control Signals ---
    input logic kernel_load,    // Assert to load kernel weights, de-assert for image processing
    input logic valid_in,       // Assert when input_column and kernel_column are valid

    // --- Data Inputs ---
    // A 12-element column from a conv_0 layer
    input logic [DATA_WIDTH-1:0] input_column [INPUT_COL_SIZE-1:0],
    // A 3-element column of the kernel to be loaded
    input logic [DATA_WIDTH-1:0] kernel_column [KERNEL_SIZE-1:0],

    // --- Data Outputs ---
    // The resulting 10-element output column
    output logic [DATA_WIDTH-1:0] output_column [PARALLEL_UNITS-1:0],
    // Asserted when the output_column data is valid
    output logic valid_out
);

    // --- Pipeline Control Logic ---

    // State machine to track the number of valid columns in the pipeline.
    typedef enum logic [2:0] {
        S_IDLE,    // 0 valid columns in the pipe.
        S_HAVE_1,  // 1 valid column in the pipe.
        S_HAVE_2,   // 2 valid columns in the pipe
        S_HAVE_3,   // 3 valid columns in the pipe, ready for the 4th to start computation.
        S_HAVE_Full   // 4 uncomputed columns
    } state_t;

    state_t current_state, next_state;
    logic   start_of_computation;
    logic   computation_started_d1;
    logic   valid_out_conv;

    // State Register (Sequential)
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= S_IDLE;
        end else begin
            current_state <= next_state;
        end
    end

    // Next State Logic (Combinational)
    always_comb begin
        next_state = current_state; // Default: remain in the current state
        if (valid_in && !kernel_load) begin
            case (current_state)
                S_IDLE:   next_state = S_HAVE_1;
                S_HAVE_1: next_state = S_HAVE_2;
                S_HAVE_2: next_state = S_HAVE_3; 
                S_HAVE_3: next_state = S_HAVE_Full;// Stay in ready state once pipeline is primed
            endcase
        end
    end

    // A computation is triggered only when the pipeline is ready (has 2 columns)
    // and the 3rd valid column arrives.
    always_ff @(negedge clk or posedge rst) begin
        if (rst) begin
            start_of_computation <= 1'b0;
        end else begin
            start_of_computation <= (current_state == S_HAVE_Full) && valid_in && !kernel_load;
        end
    end
    // Delay the trigger to align with the conv_4 internal pipeline.
    // The final result requires a 2-cycle delay from the start of computation.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            computation_started_d1 <= 1'b0;
            valid_out_conv         <= 1'b0;
        end else begin
            valid_out_conv         <= computation_started_d1;
            computation_started_d1 <= start_of_computation;
            
        end
    end

    // The layer's final output valid signal has the same timing as the internal one.
    assign valid_out = valid_out_conv;


    // --- Parallel Convolution Units ---
    // Use a generate block to create 10 instances of the conv_3 module.
    genvar i;
    generate
        for (i = 0; i < PARALLEL_UNITS; i++) begin : gen_conv_units
            
            // Temporary wire to hold the 3 input data elements for each conv_3 instance.
            // This selects the appropriate 3-element slice from the input column.
            wire [DATA_WIDTH-1:0] conv_data_slice [KERNEL_SIZE-1:0];
            assign conv_data_slice[0] = input_column[i];
            assign conv_data_slice[1] = input_column[i+1];
            assign conv_data_slice[2] = input_column[i+2];
            assign conv_data_slice[3] = input_column[i+3];
            wire [DATA_WIDTH-1:0] muxed_data_in [KERNEL_SIZE-1:0];
        
            // Use a second generate loop to create the element-wise multiplexers.
            genvar j;
            for (j = 0; j < KERNEL_SIZE; j = j + 1) begin : gen_mux
                assign muxed_data_in[j] = kernel_load ? kernel_column[j] : conv_data_slice[j];
            end

            // Instantiate the 3x3 convolution filter
            conv_4 #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE)
            ) conv_inst (
                .clk(clk),
                .rst(rst),

                // When loading kernel, all instances get the same kernel column.
                // When processing, each instance gets its unique slice of the image.
                .data_in(muxed_data_in),
                .kernel_load(kernel_load),
                .valid_in(valid_in),
                .valid_out(computation_started_d1), // Controlled by the layer's controller

                .data_out(output_column[i]) // Each instance writes to one element of the output column
            );
        end
    endgenerate

endmodule
