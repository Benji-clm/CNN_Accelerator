module fmap_capture_256 #(
    parameter int PIX_H     = 24,
    parameter       BASE_ADDR = 12'h000
)(
    input  logic            clk,
    input  logic            rst,
    input  logic            valid_col,
    input  logic [15:0]     data_col [PIX_H-1:0],

    output logic [11:0]     bram_addr_a,
    output logic [255:0]    bram_wrdata_a,
    output logic [3:0]      bram_we_a,
    
    output logic            write_done
);


// FP16 to 8-bit grayscale conversion function
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
            end else if (actual_exp >= 0) begin
                // Positive values >= 1.0 -> scale mantissa
                if (actual_exp >= 8) begin
                    result = 8'hFF; // Large values -> 255
                end else begin
                    result = mantissa[10:3]; // Take upper 8 bits of mantissa
                end
            end else begin
                // Values < 1.0 -> scale down
                if (actual_exp <= -8) begin
                    result = 8'h00;
                end else begin
                    result = mantissa[10:3] >> (-actual_exp);
                end
            end
        end
        
        fp16_to_gray = result;
    end
endfunction

logic [7:0] gray_pixels [PIX_H-1:0];
// Parallel conversion of entire column
genvar i;
generate
    for (i = 0; i < PIX_H; i++) begin : gen_parallel_conversion
        always_comb begin
            gray_pixels[i] = fp16_to_gray(data_col[i]);
        end
    end
endgenerate

// put all the gray scale into one signal (Might waste compute when module is not being driven? I added valid col just in case - actually I deleted it now due to timing issues lol)
// just assign gray concat to the output when needed

logic [255:0] gray_concat;

always_comb begin
    gray_concat = '0;
    for (int j = 0; j < PIX_H; j++) begin
        gray_concat[j*8 +: 8] = gray_pixels[j];
    end
end


logic [$clog2(PIX_H):0] col_idx;
logic [$clog2(100):0]   time_out_counter;

typedef enum logic {VALID, TIME_OUT} st_t;
st_t st;


always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        time_out_counter <= '0;
        st <= VALID;
        col_idx      <= '0;
        bram_we_a    <= 4'b0000;
        write_done   <= 1'b0;
    end
    else begin

        case(st)
            VALID: begin
                bram_we_a  <= 4'b0000;
                write_done <= 1'b0;

                if (valid_col) begin
                    if(col_idx < PIX_H-1) begin
                        st <= VALID;
                        bram_addr_a   <= BASE_ADDR + col_idx;
                        bram_wrdata_a <= gray_concat;
                        bram_we_a     <= 4'b1111;
                        col_idx <= col_idx + 1;
                    end else begin
                        bram_addr_a   <= BASE_ADDR + col_idx;
                        bram_wrdata_a <= gray_concat;
                        bram_we_a     <= 4'b1111;
                        write_done <= 1'b1;
                        col_idx    <= '0;
                        time_out_counter <= '0;
                        st <= TIME_OUT;
                    end
                end
            end
            TIME_OUT: begin
                bram_we_a  <= 4'b0000;
                write_done <= 1'b0;

                if (time_out_counter == 99) begin
                    st <= VALID;
                    time_out_counter <= '0;
                end else begin
                    time_out_counter <= time_out_counter + 1;
                end
            end
        endcase
        end
end

endmodule
