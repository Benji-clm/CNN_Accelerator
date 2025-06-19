module fixed_to_fp16 (
    input  logic signed [31:0] fixed_in,
    output logic [15:0] fp16_out
);

    logic sign;
    logic [31:0] abs_fixed_in;

    assign sign = fixed_in[31];
    assign abs_fixed_in = sign ? -fixed_in : fixed_in;

    integer msb_position;
    always_comb begin
        msb_position = -1;
        for (int i = 31; i >= 0; i--) begin
            if (abs_fixed_in[i]) begin
                msb_position = i;
                break;
            end
        end
    end

    logic [4:0] exponent;
    logic [9:0] mantissa;
    logic [31:0] shifted;

    always_comb begin
        if (fixed_in == 0) begin
            fp16_out = 16'b0;
        end else if (msb_position == -1) begin
            fp16_out = 16'b0;
        end else begin
            exponent = msb_position - 16 + 15; // e = p - 16, biased = p - 1
            shifted = abs_fixed_in << (31 - msb_position); // Leading 1 at bit 31
            mantissa = shifted[30:21]; // Bits after implicit 1

            if (exponent > 5'b11110) begin
                fp16_out = {sign, 5'b11111, 10'b0}; // Infinity
            end else if (exponent < 1) begin
                fp16_out = {sign, 5'b00000, 10'b0}; // Zero
            end else begin
                fp16_out = {sign, exponent, mantissa};
            end
        end
    end
endmodule
