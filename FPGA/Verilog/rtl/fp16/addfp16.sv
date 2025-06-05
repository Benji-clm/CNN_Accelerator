module addfp16 (
    input logic [15:0] a,      // Input data A
    input logic [15:0] b,      // Input data B
    output logic [15:0] sum     // suming sum output
);
    // Declare variables
    logic        sign_a, sign_b, sum_sign;
    logic [4:0]  exp_a, exp_b, sum_exp;  
    logic [9:0]  a_frac, b_frac; 
    logic [10:0] frac_a, frac_b;           // 1 hidden + 10 fraction bits
    logic [11:0] aligned_frac_a, aligned_frac_b; // With extra bit for carry
    
    // Additional variables for addition
    logic [12:0] sum_frac;                 // 13 bits for addition sum (with carry)
    logic [4:0]  exp_diff;                 // Exponent difference for alignment
    logic        a_larger;                 // Whether A has larger magnitude
    logic [4:0]  leading_zeros;            // For normalization
    logic [11:0] abs_diff;                 // Absolute difference for subtraction
    logic [12:0] shifted_frac;             // For storing shifted fractions
    
    // Special case flags
    logic a_is_zero, b_is_zero, a_is_inf, b_is_inf, a_is_nan, b_is_nan;

    always_comb begin
        // Default assignments
        sum = 0;
        sum_frac = 0;
        aligned_frac_a = 0;
        aligned_frac_b = 0;
        sum_exp = 0;
        sum_sign = 0;
        abs_diff = 0;
        leading_zeros = 0;
        shifted_frac = 0;

        // Extract fields
        sign_a = a[15];
        sign_b = b[15];
        exp_a = a[14:10];
        exp_b = b[14:10];
        a_frac = a[9:0];
        b_frac = b[9:0];

        // Check for special cases
        a_is_zero = (exp_a == 0) && (a_frac == 0);
        b_is_zero = (exp_b == 0) && (b_frac == 0);
        a_is_inf = (exp_a == 5'b11111) && (a_frac == 0);
        b_is_inf = (exp_b == 5'b11111) && (b_frac == 0);
        a_is_nan = (exp_a == 5'b11111) && (a_frac != 0);
        b_is_nan = (exp_b == 5'b11111) && (b_frac != 0);

        // Handle special cases
        if (a_is_nan || b_is_nan) begin
            // Return NaN
            sum = 16'h7E00; // Canonical NaN
        end else if (a_is_inf && b_is_inf) begin
            if (sign_a == sign_b) begin
                // Infinity + Infinity = Infinity (same sign)
                sum = {sign_a, 5'b11111, 10'b0000000000};
            end else begin
                // Infinity - Infinity = NaN
                sum = 16'h7E00; // Canonical NaN
            end
        end else if (a_is_inf) begin
            // Infinity + anything = Infinity
            sum = {sign_a, 5'b11111, 10'b0000000000};
        end else if (b_is_inf) begin
            // anything + Infinity = Infinity
            sum = {sign_b, 5'b11111, 10'b0000000000};
        end else if (a_is_zero && b_is_zero) begin
            // Zero + Zero = Zero (with sign handling)
            sum = (sign_a == sign_b) ? {sign_a, 15'b0} : 16'b0; // +0 for different signs
        end else if (a_is_zero) begin
            // 0 + b = b
            sum = b;
        end else if (b_is_zero) begin
            // a + 0 = a
            sum = a;
        end else begin
            // Normal addition case
            
            // Include implicit leading bit (or 0 for subnormals)
            frac_a = (exp_a == 0) ? {1'b0, a_frac} : {1'b1, a_frac};
            frac_b = (exp_b == 0) ? {1'b0, b_frac} : {1'b1, b_frac};
            
            // Determine which number has larger magnitude
            if (exp_a > exp_b || (exp_a == exp_b && frac_a >= frac_b)) begin
                a_larger = 1;
                exp_diff = exp_a - exp_b;
                sum_exp = exp_a;
            end else begin
                a_larger = 0;
                exp_diff = exp_b - exp_a;
                sum_exp = exp_b;
            end
            
            // Align significands (shift smaller number right)
            if (a_larger) begin
                aligned_frac_a = {frac_a, 1'b0};
                aligned_frac_b = (exp_diff >= 12) ? 0 : ({frac_b, 1'b0} >> exp_diff);
            end else begin
                aligned_frac_a = (exp_diff >= 12) ? 0 : ({frac_a, 1'b0} >> exp_diff);
                aligned_frac_b = {frac_b, 1'b0};
            end
            
            // Add or subtract based on signs
            if (sign_a == sign_b) begin
                // Addition (same signs)
                sum_frac = aligned_frac_a + aligned_frac_b;
                sum_sign = sign_a;
            end else begin
                // Subtraction (different signs)
                if (a_larger) begin
                    sum_frac = aligned_frac_a - aligned_frac_b;
                    sum_sign = sign_a;
                end else begin
                    sum_frac = aligned_frac_b - aligned_frac_a;
                    sum_sign = sign_b;
                end
            end
            
            // Normalize sum
            if (sum_frac == 0) begin
                // sum is zero
                sum = 16'b0; // +0
            end else if (sum_frac[12]) begin
                // Overflow in fraction - shift right and increment exponent
                sum_exp = sum_exp + 1;
                
                // Check for overflow to infinity
                if (sum_exp >= 31) begin
                    sum = {sum_sign, 5'b11111, 10'b0000000000}; // Infinity
                end else begin
                    sum = {sum_sign, sum_exp, sum_frac[11:2]};
                end
            end else begin
                // Find leading 1 for normalization
                if (sum_frac[11])      leading_zeros = 0;
                else if (sum_frac[10]) leading_zeros = 1;
                else if (sum_frac[9])  leading_zeros = 2;
                else if (sum_frac[8])  leading_zeros = 3;
                else if (sum_frac[7])  leading_zeros = 4;
                else if (sum_frac[6])  leading_zeros = 5;
                else if (sum_frac[5])  leading_zeros = 6;
                else if (sum_frac[4])  leading_zeros = 7;
                else if (sum_frac[3])  leading_zeros = 8;
                else if (sum_frac[2])  leading_zeros = 9;
                else if (sum_frac[1])  leading_zeros = 10;
                else if (sum_frac[0])  leading_zeros = 11;
                else                   leading_zeros = 12;
                
                if (leading_zeros == 0) begin
                    // Already normalized
                    sum = {sum_sign, sum_exp, sum_frac[10:1]};
                end else if (leading_zeros < sum_exp) begin
                    // Shift left to normalize and adjust exponent
                    sum_exp = sum_exp - leading_zeros;
                    
                    // Perform the shift first, then select bits
                    shifted_frac = sum_frac << leading_zeros;
                    sum = {sum_sign, sum_exp, shifted_frac[10:1]};
                end else if (leading_zeros == sum_exp) begin
                    // sum becomes subnormal
                    shifted_frac = sum_frac << leading_zeros;
                    sum = {sum_sign, 5'b00000, shifted_frac[10:1]};
                end else begin
                    // sum rounds to zero or subnormal with right-shift
                    if (sum_exp == 0) begin
                        // Already subnormal, just use bits we have
                        sum = {sum_sign, 5'b00000, sum_frac[10:1]};
                    end else begin
                        // Shift as much as we can and make subnormal
                        shifted_frac = sum_frac << sum_exp;
                        sum = {sum_sign, 5'b00000, shifted_frac[10:1]};
                    end
                end
            end
        end
    end
endmodule
