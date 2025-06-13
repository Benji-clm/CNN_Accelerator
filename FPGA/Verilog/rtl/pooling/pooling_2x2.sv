module pooling_2x2 (
    input logic [15:0] a,
    input logic [15:0] b,
    input logic store,
    input logic clk,
    input logic rst,
    output logic [15:0] pooled_value
);

    logic [15:0] max_temp;       // Stores intermediate max of the first two inputs
    logic [15:0] max_ab;         // Stores the max of the current two inputs
    logic [15:0] final_max;      // Combinational result of the final comparison

    // Instantiate comparator for the current inputs 'a' and 'b'
    fp16_max_comparator comp_ab (
        .a(a),
        .b(b),
        .max_val(max_ab)
    );

    // Instantiate comparator for the final max value
    fp16_max_comparator comp_final (
        .a(max_ab),
        .b(max_temp),
        .max_val(final_max) // Output to an intermediate wire
    );

    // Synchronous logic for all state changes
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            max_temp <= 16'h0000;
            pooled_value <= 16'h0000; // Reset the output as well
        end else begin
            if (store) begin
                max_temp <= max_ab;
            end
            pooled_value <= final_max;
        end
    end

endmodule
