module addfp16 (
    input  logic [15:0] a,    // FP16 input A
    input  logic [15:0] b,    // FP16 input B
    output logic [15:0] sum   // FP16 output sum
);

    // Unpack fields
    logic        sign_a, sign_b;
    logic [4:0]  exp_a, exp_b;  
    logic [10:0] frac_a, frac_b; // 1 hidden + 10 fraction bits
    logic [11:0] aligned_frac_a, aligned_frac_b;
    logic [4:0]  exp_diff;
    logic [12:0] sum_frac;
    logic [4:0]  sum_exp;
    logic        sum_sign;
    logic        a_bigger;

    // Unpack a
    assign sign_a = a[15];
    assign exp_a  = a[14:10];
    assign frac_a = (exp_a == 0) ? {1'b0, a[9:0]} : {1'b1, a[9:0]}; // denormals
    // Unpack b
    assign sign_b = b[15];
    assign exp_b  = b[14:10];
    assign frac_b = (exp_b == 0) ? {1'b0, b[9:0]} : {1'b1, b[9:0]}; // denormals

    // Align exponents
    assign a_bigger = (exp_a > exp_b) || ((exp_a == exp_b) && (frac_a >= frac_b));
    assign exp_diff = a_bigger ? (exp_a - exp_b) : (exp_b - exp_a);

    assign aligned_frac_a = a_bigger ? frac_a : (frac_a >> exp_diff);
    assign aligned_frac_b = a_bigger ? (frac_b >> exp_diff) : frac_b;

    // Add/subtract significands
    always_comb begin
        if (sign_a == sign_b) begin
            sum_frac = aligned_frac_a + aligned_frac_b;
            sum_sign = sign_a;
        end else begin
            if (aligned_frac_a >= aligned_frac_b) begin
                sum_frac = aligned_frac_a - aligned_frac_b;
                sum_sign = a_bigger ? sign_a : sign_b;
            end else begin
                sum_frac = aligned_frac_b - aligned_frac_a;
                sum_sign = a_bigger ? sign_b : sign_a;
            end
        end
    end

    // Normalize result
    logic [4:0] norm_shift;
    logic [10:0] norm_frac;
    logic [4:0] norm_exp;
    always_comb begin
        norm_exp = a_bigger ? exp_a : exp_b;
        if (sum_frac[12]) begin
            // Overflow, shift right
            norm_frac = sum_frac[12:2];
            norm_exp = norm_exp + 1;
        end else begin
            // Normalize left
            norm_shift = 0;
            for (int i = 11; i >= 0; i--) begin
                if (sum_frac[i]) begin
                    norm_shift = 11 - i;
                    break;
                end
            end
            norm_frac = sum_frac[11:1] << norm_shift;
            norm_exp = (norm_exp > norm_shift) ? (norm_exp - norm_shift) : 0;
        end
    end

    // Handle special cases (zero, inf, NaN)
    always_comb begin
        if (exp_a == 5'h1F) begin
            // a is Inf or NaN
            sum = a;
        end else if (exp_b == 5'h1F) begin
            // b is Inf or NaN
            sum = b;
        end else if ((exp_a == 0 && frac_a[9:0] == 0) && (exp_b == 0 && frac_b[9:0] == 0)) begin
            // both zero
            sum = 16'b0;
        end else if (norm_exp == 0) begin
            // underflow to zero
            sum = {sum_sign, 5'b0, 10'b0};
        end else if (norm_exp >= 31) begin
            // overflow to Inf
            sum = {sum_sign, 5'h1F, 10'b0};
        end else begin
            // Normal case
            sum = {sum_sign, norm_exp[4:0], norm_frac[9:0]};
        end
    end

endmodule
