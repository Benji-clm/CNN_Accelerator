module mulfp16 (
    input logic [15:0] a_in,      // Input data A
    input logic [15:0] b_in,      // Input data B
    output logic [15:0] c_out     // Resulting partial sum output
);
    // Declare variables
    logic        sign_a, sign_b;
    logic [4:0]  exp_a, exp_b;  
    logic [9:0]  a_frac, b_frac; 
    logic [10:0] frac_a, frac_b; // 1 hidden + 10 fraction bits
    logic [4:0]  product_exp;
    logic        product_sign;
    logic signed [5:0] subnormal_shift;
    
    // Additional variables for multiplication
    logic [21:0] P;              // 22-bit product of significands
    logic [4:0]  exp_adjust;     // Normalization adjustment
    logic signed [6:0] temp_exp; // Temporary exponent with extra bit for overflow
    logic [21:0] P_shifted;
    logic [21:0] P_denorm;

    always_comb begin
        // Default assignments to avoid latches
        P_shifted = 0;
        exp_adjust = 0;
        product_exp = 0;
        c_out = 0;
        P_denorm = 0;
        subnormal_shift = 0;

        // Extract sign, exponent, and fraction
        sign_a = a_in[15];
        sign_b = b_in[15];
        product_sign = sign_a ^ sign_b;

        exp_a = a_in[14:10];
        exp_b = b_in[14:10];
        a_frac = a_in[9:0];
        b_frac = b_in[9:0];

        // Handle subnormal numbers by adding implicit leading bit
        frac_a = (exp_a == 0) ? {1'b0, a_frac} : {1'b1, a_frac};
        frac_b = (exp_b == 0) ? {1'b0, b_frac} : {1'b1, b_frac};
        if (~frac_a[10]) exp_a = 1;
        if (~frac_b[10]) exp_b = 1; 
        /*
        $display("frac_a: %b", frac_a);
        $display("exp_a: %b", exp_a);
        $display("frac_b: %b", frac_b);
        $display("exp_b: %b", exp_b);
        */
        // Multiply significands and compute initial exponent
        P = frac_a * frac_b;
        temp_exp = exp_a + exp_b - 15;
        /*
        $display("P: %b", P);
        $display("exp_adjust: %b", exp_adjust);
        */
        if (P == 0) begin
            c_out = {product_sign, 5'b00000, 10'b0000000000}; // Zero
        end else begin
            // Normalize the product
            if (P[21]) begin
                P_shifted = P >> 1;
                temp_exp = temp_exp + 1;
            end else begin
                // Find leading one using a for loop (synthesizable)
                for (int i = 20; i >= 0; i--) begin
                    if (P[i]) begin
                        exp_adjust = 5'(20 - i);
                        P_shifted = P << exp_adjust;
                        temp_exp = temp_exp - exp_adjust;
                        break;
                    end
                end
            end
            /*
            $display("P_shifted: %b", P_shifted);
            $display("exp_adjust: %b", exp_adjust);
            */
            // Determine output based on exponent
            if (temp_exp >= 1 && temp_exp <= 30) begin
                // Normalized number
                product_exp = temp_exp[4:0];
                c_out = {product_sign, product_exp, P_shifted[19:10]};
            end else if (temp_exp > 30) begin
                // Overflow to infinity
                c_out = {product_sign, 5'b11111, 10'b0000000000};
            end else begin
                // Subnormal or zero (temp_exp <= 0)
                subnormal_shift = 1 - temp_exp;
                //$display("temp_exp: %b", temp_exp);
                //$display("subnormal_shift: %b", subnormal_shift);
                if (subnormal_shift > 20) begin
                    // Underflow to zero
                    c_out = {product_sign, 5'b00000, 10'b0000000000};
                end else begin
                    // Subnormal number
                    P_denorm = P_shifted >> subnormal_shift;
                    //$display("P_denorm: %b", P_denorm);
                    c_out = {product_sign, 5'b00000, P_denorm[19:10]};
                end
            end
        end
    end
endmodule
