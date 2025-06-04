// Processing Elements
module mulfp16 (
    //Control signals
    input logic clk,
    input logic rst,
    input logic load_b,
    input logic mode_fp16,        // Mode: 1 for 16-bit floating point, 0 for 8-bit integer
    input logic signed_mode,      // Signed/Unsigned mode: 1 for signed, 0 for unsigned
    // Systolic array inputs
    input logic [15:0] a_in,      // Input data A
    input logic [15:0] c_in,      // Partial sum input
    //Preloaded Weight
    input logic [15:0] b_in,      // Input data B

    //Systolica array outputs
    output logic [15:0] a_out,    // Propagated A
    output logic [15:0] c_out     // Resulting partial sum output
);
    logic [15:0] product;         // Product of A and B
    logic sign_a, sign_w, sign_result;
    logic [7:0] exp_a, exp_w, exp_result;
    logic [7:0] mant_a, mant_w, mant_c, mant_p;
    logic [15:0] mant_result;

    logic [7:0] a_int;            // Unsigned 8-bit integer version of A
    logic [7:0] w_int;            // Unsigned 8-bit integer version of B

    logic sign_c, sign_sum;
    logic [7:0] exp_c, exp_sum;
    logic [8:0] mant_sum;  // 1 extra bit for possible carry
    logic [8:0] mant_c_shifted, mant_p_shifted;
    logic [7:0] exp_diff;

    logic [15:0] sum;

    // Multiplication logic based on mode
    always_comb begin
        if (mode_fp16) begin
            // Extract sign, exponent, and mantissa
            sign_a = a_in[15];
            exp_a = a_in[14:7];
            mant_a = {1'b1, a_in[6:0]}; // Add implicit leading 1

            sign_c = c_in[15];
            exp_c  = c_in[14:7];
            mant_c = {1'b1, c_in[6:0]};

            //Compute result sign
            sign_result = sign_a ^ sign_w;
            // Add exponents and subtract bias
            exp_result = exp_a + exp_w - 8'd127;
            // Multiply mantissas
            mant_result = mant_a * mant_w;
            // Normalize the result
            if (mant_result[15]) begin
                mant_result = mant_result >> 1; // Right shift if overflow
                exp_result = exp_result + 1;    // Increment exponent
            end
            mant_p = mant_result[14:7];

            if (exp_c == exp_result) begin
                exp_sum = exp_c;
                mant_c_shifted = {1'b0, mant_c}; 
                mant_p_shifted = {1'b0, mant_p};
            end
            else if (exp_c > exp_result) begin
                exp_diff = exp_c - exp_result;
                exp_sum  = exp_c;
                mant_c_shifted = {1'b0, mant_c};
                // shift mant_p right by exp_diff
                mant_p_shifted = {1'b0, mant_p >> exp_diff};  
            end
            else begin
                exp_diff = exp_result - exp_c;
                exp_sum  = exp_result;
                mant_p_shifted = {1'b0, mant_p};
                // shift mant_c right by exp_diff
                mant_c_shifted = {1'b0, mant_c >> exp_diff};  
            end
            if (sign_c == sign_result) begin
                // Same sign -> we do an add
                mant_sum = mant_c_shifted + mant_p_shifted;
                sign_sum = sign_c;
            end
            else begin
                // Different signs -> effectively a subtraction
                if (mant_c_shifted > mant_p_shifted) begin
                    mant_sum = mant_c_shifted - mant_p_shifted;
                    sign_sum = sign_c;
                end
                else begin
                    mant_sum = mant_p_shifted - mant_c_shifted;
                    sign_sum = sign_result;
                end
            end
            // Overflow
            if (mant_sum[8] == 1'b1) begin
                mant_sum = mant_sum >> 1;
                exp_sum  = exp_sum + 1;
            end 
            else begin
                while (mant_sum[7] == 1'b0 && exp_sum > 0) begin
                    mant_sum = mant_sum << 1;
                    exp_sum  -= 1;
                end
            end
            sum = {sign_sum, exp_sum, mant_sum[6:0]};
        end 
        else begin
            if (signed_mode) begin
                // 8-bit signed integer multiplication
                a_int = $signed(a_in[7:0]);
            end else begin
                // 8-bit unsigned integer multiplication
                a_int = $unsigned(a_in[7:0]);
            end
            product = a_int * w_int;
            sum = c_in + product;
        end
    end

    // Register logic for pipelined operation
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            a_out <= 16'd0;
            c_out <= 16'd0;
        end
        else if (load_b)begin
            if (mode_fp16) begin
                sign_w <= b_in[15];
                exp_w <= b_in[14:7];
                mant_w <= {1'b1, b_in[6:0]}; 
            end
            else w_int <= signed_mode ? $signed(b_in[7:0]) : $unsigned(b_in[7:0]);
        end 
        else begin
            a_out <= a_in;

            c_out <= sum;
        end
    end

endmodule
