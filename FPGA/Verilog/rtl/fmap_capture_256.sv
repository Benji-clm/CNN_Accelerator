module fmap_capture_256 #(
    parameter int PIX_H     = 24,
    parameter       BASE_ADDR = 12'h000
)(
    input  logic            clk,
    input  logic            rst,
    input  logic            valid_col,
    input  logic [15:0]     data_col [PIX_H-1:0],

    output logic [11:0]     bram_addr_a,
    output logic [191:0]    bram_wrdata_a, // FIXED: Width changed to 192 bits (24 * 8)
    output logic            bram_we_a,
    
    output logic            write_done
);

// FP16 to 8-bit grayscale conversion function - UNCOMMENTED AND SAFE
function automatic [7:0] fp16_to_gray;
    input [15:0] fp16_val;
    
    reg sign;
    reg [4:0] exponent;
    reg [10:0] mantissa;
    reg [7:0] result;
    
    begin
        sign = fp16_val[15];
        exponent = fp16_val[14:10];
        mantissa = {1'b1, fp16_val[9:0]}; // Add implicit leading 1
        
        // Handle special cases
        if (exponent == 5'b00000) begin
            // Zero or denormalized
            result = 8'h00;
        end else if (exponent == 5'b11111) begin
            // Infinity or NaN
            result = 8'hFF;
        end else begin
            // Normalized number
            // Bias is 15 for fp16, so exponent - 15 gives actual exponent
            reg signed [5:0] actual_exp;
            actual_exp = exponent - 5'd15;
            
            if (sign) begin
                result = 8'h00; // Negative values -> 0
            end else if (actual_exp >= 8) begin
                result = 8'hFF; // Large values -> 255
            end else if (actual_exp >= 0) begin
                // Positive values >= 1.0 -> scale mantissa
                result = mantissa[10:3]; // Take upper 8 bits of mantissa
            end else begin
                // Values < 1.0 -> scale down
                if (actual_exp <= -8) begin
                    result = 8'h00;
                end else begin
                    result = mantissa >> (-actual_exp);
                end
            end
        end
        fp16_to_gray = result;
    end
endfunction

// Parallel conversion of FP16 to grayscale
logic [7:0] gray_pixels [PIX_H-1:0];
genvar i;
generate
    for (i = 0; i < PIX_H; i++) begin : gen_parallel_conversion
        always_comb begin
            gray_pixels[i] = fp16_to_gray(data_col[i]);
        end
    end
endgenerate

// Concatenate all grayscale pixels into a 192-bit word
logic [191:0] gray_concat; // FIXED: Width changed to 192
always_comb begin
    // No need to pre-assign to '0' if every bit is assigned
    for (int j = 0; j < PIX_H; j++) begin
        gray_concat[j*8 +: 8] = gray_pixels[j];
    end
end

// Column counter and control logic
logic [$clog2(PIX_H):0] col_idx;
logic [1:0] write_counter;  // 2-bit counter for write cycles

// Write control logic - SINGLE always_ff block
always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        write_counter <= 2'b00;
        bram_we_a <= 1'b0;
        col_idx <= '0;
        bram_addr_a <= BASE_ADDR;
        bram_wrdata_a <= '0;
        write_done <= 1'b0;
    // NOTE: Consider if you need a conv_done reset condition here too
    end else begin
        // Default behavior: keep write_done low unless explicitly set
        if (write_done) begin
            write_done <= 1'b0;
        end

        case (write_counter)
            2'b00: begin  // IDLE state
                if (valid_col) begin
                    // Start write sequence
                    bram_addr_a <= BASE_ADDR + col_idx;
                    bram_wrdata_a <= gray_concat;
                    bram_we_a <= 1'b1;
                    write_counter <= 2'b01;
                end
            end
            
            2'b01: begin  // CYCLE 1 - Continue write
                // Data and address are held from previous state
                bram_we_a <= 1'b1;
                write_counter <= 2'b10;
            end
            
            2'b10: begin  // CYCLE 2 - Finish write
                bram_we_a <= 1'b0;
                write_counter <= 2'b00;
                
                if (col_idx >= PIX_H-1) begin
                    write_done <= 1'b1; // Signal completion
                    col_idx <= '0;      // Reset for next frame
                end else begin
                    col_idx <= col_idx + 1;
                end
            end
            
            default: begin
                write_counter <= 2'b00;
                bram_we_a <= 1'b0;
            end
        endcase
    end
end

endmodule
