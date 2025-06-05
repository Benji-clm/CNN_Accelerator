module reduction #(
    parameter data_width = 16,
    parameter mat_width = 2
)(
    input logic                         clk,
    input logic                         rst,
    input logic                         valin_in,
    input logic [mat_width-1:0] column [data_width-1:0],
    output logic                        valid_out,
    output logic [data_width-1:0]       sum
);

reg [data_width-1:0] val_1;
reg [data_width-1:0] val_2;
reg [data_width-1:0] val_3;
reg [data_width-1:0] val_4;

reg col_n = 0;

reg output_ready = 0;

always_ff @(posedge clk) begin
    if (rst) begin
        sum <= 0;
        output_ready <= 0;
        col_n <= 0;
        val_1 <= 0;
        val_2 <= 0;
        val_3 <= 0;
        val_4 <= 0;
    end else if (valin_in && !output_ready) begin
            if(col_n == 1'b0) begin 
                val_1 <= column[0];
                val_2 <= column[1];
                col_n <= 1'b1;
            end else begin
                val_3 <= column[0];
                val_4 <= column[1];
                col_n <= 1'b0;
                sum <= (val_1 + val_2 + val_3 + val_4); // Compute the sum of the four values
                valid_out <= 1; // Signal that the sum is ready
            end
    end else if (output_ready) begin
        valid_out <= 0;
    end
end

endmodule
