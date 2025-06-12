module pooling_2x2 (
    input logic [15:0] a,
    input logic [15:0] b,
    input logic store, 
    input logic clk,
    input logic rst,
    output logic [15:0] pooled_value
);

    logic [15:0] max_temp;        // Stores intermediate max
    // Instantiate comparators
    logic [15:0] max_ab;
    fp16_max_comparator comp_ab (
        .a(a),
        .b(b),
        .max_val(max_ab)
    );

    fp16_max_comparator comp_final (
        .a(max_ab),
        .b(max_temp),
        .max_val(pooled_value)
    );

    // Synchronous logic
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            max_temp <= 16'h0000;
        end else if (store) begin
            max_temp <= max_ab;
        end
    end
endmodule
