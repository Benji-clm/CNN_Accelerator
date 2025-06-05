module ieee_to_hex (
    input               clk,
    input               rst,
    input wire [15:0]   float_in,
    output reg [15:0]   int_out
);

reg [15:0] next_val;
reg [4:0] exponent;
reg [9:0] mantissa;

always @(*) begin 
    exponent = float_in[14:10];
    if(exponent == 5'h0 || exponent == 5'h1F || exponent < 5'd15) next_val = 16'b0;
    else begin
        exponent = exponent - 5'd15;
        mantissa = float_in[9:0];
        next_val = {5'b0, 1'b1, mantissa};
        // $display("(bin): %b", next_val);
        if(exponent <= 5'd10) begin
            next_val = next_val >> (5'd10 - exponent);
        end else begin
            next_val = next_val << (exponent - 5'd10);
        end
        // $display("(bin): %b", next_val);
    end
end

always @(posedge clk) begin 
    if(rst) int_out <= 16'b0;
    else int_out <= next_val;
end

endmodule
