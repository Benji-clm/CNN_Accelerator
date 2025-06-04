module reLU #(
    parameter DATA_WIDTH = 16,

)(
    input logic clk,
    input logic rst,
    input logic [DATA_WIDTH-1:0] data_in,
    output logic [DATA_WIDTH-1:0] data_out
);

    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            data_out <= 0;
        end else begin
            // Apply ReLU activation function
            if (data_in[DATA_WIDTH-1] == 1'b0) begin
                data_out <= data_in; // Positive value, pass through
            end else begin
                data_out <= 0; // Negative value, set to zero
            end
        end
    end


endmodule