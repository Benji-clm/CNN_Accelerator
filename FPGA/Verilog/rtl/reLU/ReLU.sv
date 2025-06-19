module ReLU (
    input logic signed [15:0] data_in,
    output logic signed [15:0] data_out
);
    always_comb begin
        // Apply ReLU activation function
        if (data_in[15] == 1'b0) begin
            data_out = data_in; // Positive value, pass through
        end else begin
            data_out = 16'b0; // Negative value, set to zero
        end
    end
endmodule
