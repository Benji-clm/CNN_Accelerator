module pooling_layer (
    input logic clk,
    input logic rst,
    input logic input_valid,
    input logic [23:0][15:0] input_row,  // One row of 24 FP16 elements
    output logic output_valid,
    output logic [11:0][15:0] output_row  // One pooled row of 12 FP16 elements
);

    logic [23:0][15:0] prev_row;  // Buffer for the previous row
    logic toggle;                 // Toggle bit to alternate between buffering and computing

    logic [11:0][15:0] pooled_values;  // Combinational output of pooling

    // Combinational logic for 2x2 max pooling
    for (genvar j = 0; j < 12; j++) begin : pool_gen
        pooling_2x2 pool(
            .a(prev_row[2*j]),
            .b(prev_row[2*j+1]),
            .c(input_row[2*j]),
            .d(input_row[2*j+1]),
            .pooled_value(pooled_values[j])
        );
    end

    // Sequential logic for buffering and output
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            prev_row <= '0;
            toggle <= 0;
            output_valid <= 0;
            output_row <= '0;
        end else if (input_valid) begin
            if (toggle) begin
                output_row <= pooled_values;  // Output the pooled row
                output_valid <= 1;           // Signal valid output
            end else begin
                prev_row <= input_row;       // Store the current row
                output_valid <= 0;           // No output yet
            end
            toggle <= ~toggle;              // Flip toggle for next cycle
        end else begin
            output_valid <= 0;              // No valid output when input is invalid
        end
    end

endmodule