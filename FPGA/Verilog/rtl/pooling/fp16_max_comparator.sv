module fp16_max_comparator (
    input logic [15:0] a,
    input logic [15:0] b,
    output logic [15:0] max_val
);
    // Extract exponent and mantissa
    logic [4:0] exp_a = a[14:10];
    logic [4:0] exp_b = b[14:10];
    logic [9:0] mant_a = a[9:0];
    logic [9:0] mant_b = b[9:0];

    always_comb begin
        if (exp_a > exp_b) begin
            max_val = a;
        end else if (exp_a < exp_b) begin
            max_val = b;
        end else begin
            // Exponents equal, compare mantissas
            max_val = (mant_a > mant_b) ? a : b;
        end
    end
endmodule
