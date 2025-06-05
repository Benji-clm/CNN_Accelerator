module pooling_2x2 (
    input logic [15:0] a, // a b
    input logic [15:0] b, // c d
    input logic [15:0] c,
    input logic [15:0] d,
    output logic [15:0] pooled_value
);
    logic [15:0] max_top;
    logic [15:0] max_bot;

    // Compare top row
    fp16_max_comparator comp_top (
        .a(a),
        .b(b),
        .max_val(max_top)
    );

    // Compare bottom row
    fp16_max_comparator comp_bot (
        .a(c),
        .b(d),
        .max_val(max_bot)
    );

    // Compare row results
    fp16_max_comparator comp_final (
        .a(max_top),
        .b(max_bot),
        .max_val(pooled_value)
    );
endmodule