// ====================================================================
//  Feature-map capture for 256-bit BRAM
//  – No handshake, captures 24 columns when valid_col is asserted
//  – Converts fp16 IEEE 754 data to 8-bit grayscale
//  – Packs 32 pixels (256 bits) per BRAM write
//  – Writes entire feature map to BRAM efficiently
// ====================================================================
module fmap_capture #(
    parameter int PIX_W     = 24,      // map width  (columns)
    parameter int PIX_H     = 24,      // map height (rows)
    parameter int BASE_ADDR = 0,       // byte offset in display BRAM
    parameter int PIX_BITS  = 8        // we store 8-bit greyscale
)(
    input  logic                      clk,
    input  logic                      rst,

    // ---------- CNN side (no handshake) ----------
    input  logic                      valid_col,               // start capture signal
    input  logic [15:0]               data_col [PIX_H-1:0],    // PIX_H rows in parallel (fp16 format)

    // ---------- display-BRAM write port (256-bit width) ----------
    output logic [11:0]               bram_addr,
    output logic [255:0]              bram_wdata,
    output logic                      bram_we,

    // ---------- completion pulse --------------
    output logic                      done                     // 1-clk when full map stored
);

    // ----------------------------------------------------------------
    // Internal state and parameters
    // ----------------------------------------------------------------
    localparam int ROWS = PIX_H;
    localparam int COLS = PIX_W;
    localparam int PIXELS_PER_WRITE = 32;              // 256 bits / 8 bits per pixel
    localparam int TOTAL_WRITES = (ROWS * COLS + PIXELS_PER_WRITE - 1) / PIXELS_PER_WRITE;

    // Feature map storage - 24x24 pixels, 8-bit each
    logic [7:0] fmap_buffer [0:ROWS-1][0:COLS-1];
    
    // Capture and write control
    logic [$clog2(COLS)-1:0]  col_capture_cnt;         // column capture counter
    logic [$clog2(TOTAL_WRITES)-1:0] write_cnt;        // BRAM write counter
    logic capture_active;
    logic write_active;

    typedef enum logic [1:0] {IDLE, CAPTURE, WRITE} st_t;
    st_t st;

    // ----------------------------------------------------------------
    // FP16 to 8-bit grayscale conversion function
    // ----------------------------------------------------------------
    function automatic logic [7:0] fp16_to_gray(logic [15:0] fp16_val);
        logic sign;
        logic [4:0] exponent;
        logic [10:0] mantissa;
        logic [7:0] result;
        
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
            // For conversion to 8-bit: scale and clamp to [0, 255]
            logic signed [5:0] actual_exp;
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
        
        return result;
    endfunction

    // ----------------------------------------------------------------
    // Main FSM
    // ----------------------------------------------------------------
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            st              <= IDLE;
            col_capture_cnt <= '0;
            write_cnt       <= '0;
            capture_active  <= 1'b0;
            write_active    <= 1'b0;
            bram_we         <= 1'b0;
            done            <= 1'b0;
        end
        else begin
            // defaults every cycle
            bram_we <= 1'b0;
            done    <= 1'b0;

            unique case (st)
            // ================================================= IDLE
            IDLE: begin
                capture_active  <= 1'b0;
                write_active    <= 1'b0;
                col_capture_cnt <= '0;
                write_cnt       <= '0;
                
                if (valid_col) begin
                    capture_active <= 1'b1;
                    st <= CAPTURE;
                end
            end

            // ========================================= CAPTURE columns
            CAPTURE: begin
                if (capture_active) begin
                    // Convert and store current column
                    for (int r = 0; r < ROWS; r++) begin
                        fmap_buffer[r][col_capture_cnt] <= fp16_to_gray(data_col[r]);
                    end
                    
                    col_capture_cnt <= col_capture_cnt + 1;
                    
                    if (col_capture_cnt == COLS-1) begin
                        // All columns captured, start writing to BRAM
                        capture_active <= 1'b0;
                        write_active   <= 1'b1;
                        write_cnt      <= '0;
                        st <= WRITE;
                    end
                end
            end

            // ========================================= WRITE to BRAM
            WRITE: begin
                if (write_active) begin
                    bram_we <= 1'b1;
                    bram_addr <= BASE_ADDR + write_cnt;
                    
                    // Pack 32 pixels into 256-bit word
                    for (int i = 0; i < PIXELS_PER_WRITE; i++) begin
                        logic [$clog2(ROWS*COLS)-1:0] pixel_idx;
                        logic [$clog2(ROWS)-1:0] row_idx;
                        logic [$clog2(COLS)-1:0] col_idx;
                        
                        pixel_idx = write_cnt * PIXELS_PER_WRITE + i;
                        
                        if (pixel_idx < ROWS * COLS) begin
                            row_idx = pixel_idx / COLS;
                            col_idx = pixel_idx % COLS;
                            bram_wdata[i*8 +: 8] <= fmap_buffer[row_idx][col_idx];
                        end else begin
                            bram_wdata[i*8 +: 8] <= 8'h00; // Padding
                        end
                    end
                    
                    write_cnt <= write_cnt + 1;
                    
                    if (write_cnt == TOTAL_WRITES-1) begin
                        // All data written
                        write_active <= 1'b0;
                        done <= 1'b1;
                        st <= IDLE;
                    end
                end
            end
            endcase
        end
    end
endmodule
