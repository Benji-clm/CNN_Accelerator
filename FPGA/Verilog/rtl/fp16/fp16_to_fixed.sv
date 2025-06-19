module fp16_to_fixed (
    input  logic [15:0] fp16_in,
    output logic signed [31:0] fixed_out
);

    logic sign;
    logic [4:0] exponent;
    logic [9:0] mantissa;

    assign sign = fp16_in[15];
    assign exponent = fp16_in[14:10];
    assign mantissa = fp16_in[9:0];

    logic [10:0] full_mantissa;
    assign full_mantissa = {1'b1, mantissa};

    integer unsigned_exponent;
    assign unsigned_exponent = exponent;

    integer shift_amount;
    logic [31:0] shifted_mantissa;

    always_comb begin
        shift_amount = unsigned_exponent - 15 + (16 - 10); // = exponent - 9

        if (exponent == 5'b00000) begin
            shifted_mantissa = 32'b0;
        end else if (exponent == 5'b11111) begin
            shifted_mantissa = sign ? 32'h80000000 : 32'h7FFFFFFF;
        end else begin
            if (shift_amount >= 0) begin
                shifted_mantissa = { {21{1'b0}}, full_mantissa } <<< shift_amount;
            end else begin
                shifted_mantissa = { {21{1'b0}}, full_mantissa } >>> -shift_amount;
            end
        end

        fixed_out = sign ? -shifted_mantissa : shifted_mantissa;
    end
endmodule