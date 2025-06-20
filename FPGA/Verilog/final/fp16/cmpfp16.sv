module cmpfp16 (
    input logic [15:0] a,      // First FP16 number
    input logic [15:0] b,      // Second FP16 number
    output logic a_gt_b,       // a > b
    output logic a_eq_b,       // a == b
    output logic a_lt_b        // a < b
);
    // Extract fields
    logic        sign_a, sign_b;
    logic [4:0]  exp_a, exp_b;
    logic [9:0]  frac_a, frac_b;
    
    // Special case flags
    logic a_is_zero, b_is_zero, a_is_nan, b_is_nan, a_is_inf, b_is_inf;
    logic both_zero, both_inf, any_nan;
    
    // Magnitude comparison flags
    logic mag_a_gt_b, mag_a_eq_b;
    
    always_comb begin
        // Extract fields
        sign_a = a[15];
        sign_b = b[15];
        exp_a = a[14:10];
        exp_b = b[14:10];
        frac_a = a[9:0];
        frac_b = b[9:0];
        
        // Check for special cases
        a_is_zero = (exp_a == 0) && (frac_a == 0);
        b_is_zero = (exp_b == 0) && (frac_b == 0);
        a_is_nan = (exp_a == 5'b11111) && (frac_a != 0);
        b_is_nan = (exp_b == 5'b11111) && (frac_b != 0);
        a_is_inf = (exp_a == 5'b11111) && (frac_a == 0);
        b_is_inf = (exp_b == 5'b11111) && (frac_b == 0);
        
        both_zero = a_is_zero && b_is_zero;
        both_inf = a_is_inf && b_is_inf;
        any_nan = a_is_nan || b_is_nan;
        
        // Magnitude comparison (ignoring sign)
        if (exp_a > exp_b) begin
            mag_a_gt_b = 1;
            mag_a_eq_b = 0;
        end else if (exp_a < exp_b) begin
            mag_a_gt_b = 0;
            mag_a_eq_b = 0;
        end else begin
            // Exponents equal, compare fractions
            if (frac_a > frac_b) begin
                mag_a_gt_b = 1;
                mag_a_eq_b = 0;
            end else if (frac_a < frac_b) begin
                mag_a_gt_b = 0;
                mag_a_eq_b = 0;
            end else begin
                mag_a_gt_b = 0;
                mag_a_eq_b = 1;
            end
        end
        
        // Determine comparison results
        if (any_nan) begin
            // NaN comparisons are always false
            a_gt_b = 0;
            a_eq_b = 0;
            a_lt_b = 0;
        end else if (both_zero) begin
            // +0 == -0
            a_gt_b = 0;
            a_eq_b = 1;
            a_lt_b = 0;
        end else if (both_inf) begin
            // Compare infinity signs
            if (sign_a == sign_b) begin
                a_gt_b = 0;
                a_eq_b = 1;
                a_lt_b = 0;
            end else if (sign_a) begin
                // a is -inf, b is +inf
                a_gt_b = 0;
                a_eq_b = 0;
                a_lt_b = 1;
            end else begin
                // a is +inf, b is -inf
                a_gt_b = 1;
                a_eq_b = 0;
                a_lt_b = 0;
            end
        end else if (a_is_inf) begin
            // a is infinity, b is finite
            if (sign_a) begin
                // a is -inf
                a_gt_b = 0;
                a_eq_b = 0;
                a_lt_b = 1;
            end else begin
                // a is +inf
                a_gt_b = 1;
                a_eq_b = 0;
                a_lt_b = 0;
            end
        end else if (b_is_inf) begin
            // b is infinity, a is finite
            if (sign_b) begin
                // b is -inf
                a_gt_b = 1;
                a_eq_b = 0;
                a_lt_b = 0;
            end else begin
                // b is +inf
                a_gt_b = 0;
                a_eq_b = 0;
                a_lt_b = 1;
            end
        end else if (a_is_zero) begin
            // a is zero, b is non-zero
            if (sign_b) begin
                // b is negative
                a_gt_b = 1;
                a_eq_b = 0;
                a_lt_b = 0;
            end else begin
                // b is positive
                a_gt_b = 0;
                a_eq_b = 0;
                a_lt_b = 1;
            end
        end else if (b_is_zero) begin
            // b is zero, a is non-zero
            if (sign_a) begin
                // a is negative
                a_gt_b = 0;
                a_eq_b = 0;
                a_lt_b = 1;
            end else begin
                // a is positive
                a_gt_b = 1;
                a_eq_b = 0;
                a_lt_b = 0;
            end
        end else begin
            // Both are finite non-zero numbers
            if (sign_a != sign_b) begin
                // Different signs
                if (sign_a) begin
                    // a is negative, b is positive
                    a_gt_b = 0;
                    a_eq_b = 0;
                    a_lt_b = 1;
                end else begin
                    // a is positive, b is negative
                    a_gt_b = 1;
                    a_eq_b = 0;
                    a_lt_b = 0;
                end
            end else begin
                // Same signs - compare magnitudes
                if (mag_a_eq_b) begin
                    a_gt_b = 0;
                    a_eq_b = 1;
                    a_lt_b = 0;
                end else if (sign_a) begin
                    // Both negative - larger magnitude means smaller value
                    a_gt_b = ~mag_a_gt_b;
                    a_eq_b = 0;
                    a_lt_b = mag_a_gt_b;
                end else begin
                    // Both positive - larger magnitude means larger value
                    a_gt_b = mag_a_gt_b;
                    a_eq_b = 0;
                    a_lt_b = ~mag_a_gt_b;
                end
            end
        end
    end
endmodule
