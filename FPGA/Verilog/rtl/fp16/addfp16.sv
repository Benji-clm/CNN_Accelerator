module addfp16 (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] sum
);
// Internal signals for fixed-point representation
    logic signed [31:0] a_fixed;
    logic signed [31:0] b_fixed;
    logic signed [31:0] sum_fixed;

    // Instantiate the FP16 to Fixed-Point converters
    fp16_to_fixed u_fp16_to_fixed_a (
        .fp16_in(a),
        .fixed_out(a_fixed)
    );

    fp16_to_fixed u_fp16_to_fixed_b (
        .fp16_in(b),
        .fixed_out(b_fixed)
    );

    // The core fixed-point addition
    // This is significantly simpler than a full FP adder
    assign sum_fixed = a_fixed + b_fixed;

    // Instantiate the Fixed-Point to FP16 converter
    fixed_to_fp16 u_fixed_to_fp16_result (
        .fixed_in(sum_fixed),
        .fp16_out(sum)
    );

endmodule
