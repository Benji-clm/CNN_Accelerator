module pooling_layer #(
    parameter DATA_WIDTH = 16,
    parameter WINDOWS    = 12  // This will be overridden by conv_layer_1 to 5
)(
    input logic clk,
    input logic rst,

    // --- Control Signals ---
    input logic valid_in, // Asserted for one cycle when an input_column is ready

    // --- Data Inputs ---
    // Note: The hardware only ever processes one column at a time.
    // The FSM determines if it's the 1st or 2nd element of a pair.
    input logic [DATA_WIDTH-1:0] input_column [WINDOWS*2-1:0],

    // --- Data Outputs ---
    output logic [DATA_WIDTH-1:0] output_column [WINDOWS-1:0],
    output logic valid_out  // Asserted for one cycle when output_column is valid
);

    // Internal state machine to process pairs of inputs
    typedef enum logic {S_IDLE, S_HAVE_ONE} state_t;
    state_t current_state;

    // Register to store the first element of each pair
    logic [DATA_WIDTH-1:0] first_element_reg [WINDOWS-1:0];
    logic                  valid_out_reg;

    // The output is purely registered for timing purposes
    assign valid_out = valid_out_reg;

    // --- State Machine and Max-Pool Logic ---

    // This single process controls the state, the data storage, and the output generation.
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= S_IDLE;
            valid_out_reg <= 1'b0;
            for (int i = 0; i < WINDOWS; i++) begin
                first_element_reg[i] <= '0;
                output_column[i]     <= '0;
            end
        end
        else begin
            // By default, de-assert valid_out on every cycle unless specified otherwise.
            valid_out_reg <= 1'b0;

            // Only process when valid data arrives
            if (valid_in) begin
                case (current_state)
                    // We have no data stored. The current input is the FIRST element of a pair.
                    S_IDLE: begin
                        // Store the incoming column data in our register.
                        // For pooling, we take pairs like (in[0], in[1]), (in[2], in[3]), etc.
                        for (int i = 0; i < WINDOWS; i++) begin
                           first_element_reg[i] <= input_column[2*i];
                        end
                        // Transition to the next state to wait for the second element.
                        current_state <= S_HAVE_ONE;
                    end

                    // We have the first element stored. The current input is the SECOND element.
                    S_HAVE_ONE: begin
                        // Perform the max operation for each of the parallel windows.
                        for (int i = 0; i < WINDOWS; i++) begin
                            logic [DATA_WIDTH-1:0] first_val  = first_element_reg[i];
                            // The second value of the pair is in the next position.
                            logic [DATA_WIDTH-1:0] second_val = input_column[2*i+1];

                            // Simple magnitude comparison (ignores sign, fine for ReLU output)
                            if (first_val[14:0] > second_val[14:0]) begin
                                output_column[i] <= first_val;
                            end else begin
                                output_column[i] <= second_val;
                            end
                        end
                        // The output is now valid for this one cycle.
                        valid_out_reg <= 1'b1;
                        // Return to idle to wait for the next pair.
                        current_state <= S_IDLE;
                    end
                endcase
            end
        end
    end

endmodule