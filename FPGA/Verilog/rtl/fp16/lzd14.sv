module lzd14 (
    input  logic [13:0] a,
    output logic [3:0]  position,
    output logic        valid
);
    wire [2:0] pUpper, pLower;
    wire vUpper, vLower;

    lzd8 lzd8_1 (.a(a[13:6]), .position(pUpper), .valid(vUpper)); 
    lzd8 lzd8_2 (.a({2'b0, a[5:0]}), .position(pLower), .valid(vLower)); 

    assign valid = vUpper | vLower; 
    assign position[3] = ~vUpper; 
    assign position[2] = ~vUpper ? pLower[2] : pUpper[2]; 
    assign position[1] = ~vUpper ? pLower[1] : pUpper[1]; 
    assign position[0] = ~vUpper ? pLower[0] : pUpper[0]; 
endmodule
