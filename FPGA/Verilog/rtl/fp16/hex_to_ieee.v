module hex_to_ieee (
    input               clk,
    input               rst,
    input wire [15:0]   int_in,
    output reg [15:0]   float_out
);
    reg [15:0]          next_val;
    int i;
    int j;
    reg [4:0] exponent;
    reg [9:0] mantissa;


    always @(*) begin
        if(int_in == 16'b0) begin
            next_val = 16'b0;
        end else begin
            i = 15;
            while(i != 0 && int_in[i] == 0) i = i - 1;
            exponent = 5'd15 + i;
            if (i > 10) begin
                mantissa = int_in[i-1 -: 10];
                // $display("i>=10");
                // $display("mantissa (hex): 0x%h", mantissa);
                // $display("mantissa (bin): %b", mantissa);
                if(int_in[i-11] == 1) begin // if first bit of mantissa is 1 then we *might* round up
                    j = i-11;
                    while(j != 0 && int_in[j] == 0) j = j - 1;
                    if(j != 0) begin // if j != 0 then the rest of the discarded mentissa is not 0s hence we round up the mantissa
                        // $display("mantissa before change (bin): %b", mantissa);
                        if(mantissa == 10'b1111111111) begin  // Will overflow
                            // $display("Overflow will occur - adjusting exponent");
                            mantissa = 10'b0000000000;
                            exponent = exponent + 1'b1;
                        end else begin
                            mantissa = mantissa + 1'b1;
                        end
                        // $display("mantissa changed (bin): %b", mantissa);
                    end
                    else begin // in the case where it is 0's (meaning exactly half-way) then the rounding depends on whether the mantissa begin even or odd
                        if(int_in[i-10] == 1) begin 
                            if(mantissa == 10'b1111111111) begin  // Will overflow
                                // $display("Overflow will occur - adjusting exponent");
                                mantissa = 10'b0000000000;
                                exponent = exponent + 1'b1;
                            end else begin
                                mantissa = mantissa + 1'b1;
                            end
                            // $display("mantissa changed (bin): %b", mantissa);
                        end
                    end
                end


            end else begin
                mantissa = int_in << (10 - i);
            end
            
            next_val = {1'b0, exponent, mantissa[9:0]};
        end
    end

    always @(posedge clk) begin 
        if(rst) float_out <= 16'b0;
        else float_out <= next_val;
    end

endmodule
