module pooling_2x2 (
    input logic signed [15:0] a,
    input logic signed [15:0] b,
    input logic store,
    input logic clk,
    input logic rst,
    output logic signed [15:0]  pooled_value
);
    localparam SIGNED_MIN = 16'h8000;
    logic signed [15:0] max_temp;       // Stores intermediate max of the first two inputs
    logic signed [15:0] max_ab;         // Stores the max of the current two inputs
    logic signed [15:0] final_max;      // Combinational result of the final comparison

    assign max_ab = (a > b) ? a : b;
    assign final_max = (max_ab > max_temp) ? max_ab : max_temp;

    // Synchronous logic for all state changes
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            max_temp     <= SIGNED_MIN;
            pooled_value <= SIGNED_MIN;
        end else begin
            if (store) begin
                max_temp <= max_ab;
            end
            pooled_value <= final_max;
        end
    end

endmodule
