module pooling_control (
    input logic clk,
    input logic rst,
    input logic valid_in,
    output logic store,
    output logic valid_out
);

    // State encoding
    localparam WAIT_FIRST = 1'b0;
    localparam WAIT_SECOND = 1'b1;

    logic current_state, next_state;

    // State register
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state <= WAIT_FIRST;
        end else begin
            current_state <= next_state;
        end
    end

    // Next state and output logic
    always_comb begin
        next_state = current_state;
        store = 1'b0;
        valid_out = 0;
        case (current_state)
            WAIT_FIRST: begin
                if (valid_in) begin
                    next_state = WAIT_SECOND;
                    store = 1'b1;
                end
            end
            WAIT_SECOND: begin
                if (valid_in) begin
                    next_state = WAIT_FIRST;
                    valid_out = 1;
                end
            end
        endcase
    end

endmodule
