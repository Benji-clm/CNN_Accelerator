module easy_image #(
    parameter int PIX_W     = 24,      // map width  (columns)
    parameter int PIX_H     = 24,      // map height (rows)
    parameter int X_SIZE = 640
)(
    input                   clk,
    input                   rst,
    input                   valid_col,               // Signal indicating new column available
    input  logic [15:0]     data_col [PIX_H-1:0],   // PIX_H rows in parallel (fp16 format)

    // No BRAM interface needed for simple column processing
    output logic [7:0]                 current_gray_pixel,     // Current grayscale pixel output
    output logic [$clog2(X_SIZE)-1:0]        x_coordinate,
    output logic                        pixel_valid              // Indicates when current_gray_pixel is valid
);



// ================================================================
// Simple Column Buffer and FP16 to Grayscale Conversion
// Captures one column and outputs pixels sequentially over 24 cycles
// ================================================================

// Column buffer for PIX_H FP16 values
logic [15:0] column_buffer [PIX_H-1:0];
logic [$clog2(X_SIZE)-1:0] output_index;       // Index for reading from buffer (0 to PIX_H-1)
logic [4:0] output_counter;     // Counter for output cycles (0 to PIX_H-1)
logic output_active;            // Signal that we're outputting pixels

// State machine states
typedef enum logic [1:0] {
    IDLE,           // Ready to capture new column
    OUTPUTTING      // Outputting pixels from captured column
} state_t;

state_t current_state;

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

// Column capture and output state machine
typedef enum logic [2:0] {
    COL_IDLE,
    COL_CAPTURE,
    COL_OUTPUT,
    COL_WRITE_BRAM
} col_state_t;

col_state_t col_state;

// Main state machine
always @(posedge clk) begin
    if (rst) begin
        current_state <= IDLE;
        output_counter <= 0;
        output_active <= 0;
        pixel_valid <= 0;
    end else begin
        // Default values
        pixel_valid <= 0;
        
        case (current_state)
            IDLE: begin
                output_active <= 0;
                
                if (valid_col) begin
                    // Capture the entire column in one cycle
                    for (int i = 0; i < PIX_H; i++) begin
                        column_buffer[i] <= data_col[i];
                    end
                    
                    // Start outputting pixels
                    current_state <= OUTPUTTING;
                    output_index <= 0;
                    output_counter <= 0;
                    output_active <= 1;
                    pixel_valid <= 1; // First pixel will be valid next cycle
                end
            end
            
            OUTPUTTING: begin
                pixel_valid <= 1; // Output is valid during this state
                
                if (output_counter == PIX_H - 1) begin
                    // Finished outputting all pixels
                    current_state <= IDLE;
                    output_active <= 0;
                    pixel_valid <= 0;
                end else begin
                    // Move to next pixel
                    output_counter <= output_counter + 1;
                    output_index <= output_index + 1;
                end
            end
            
            default: begin
                current_state <= IDLE;
            end
        endcase
    end
end

// Output pixel generation
assign current_gray_pixel = output_active ? fp16_to_gray(column_buffer[output_index]) : 8'h00;
assign x_coordinate = output_index;

endmodule
