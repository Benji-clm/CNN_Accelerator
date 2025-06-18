module lzd8 (
    input  logic [7:0] a,
    output logic [2:0] position,
    output logic       valid
);
    wire [1:0] pUpper, pLower;
    wire vUpper, vLower;

    lzd4 lzd4_1 (.a(a[7:4]), .position(pUpper), .valid(vUpper)); 
    lzd4 lzd4_2 (.a(a[3:0]), .position(pLower), .valid(vLower)); 

    assign valid = vUpper | vLower; 
    assign position[2] = ~vUpper; 
    assign position[1] = ~vUpper ? pLower[1] : pUpper[1]; 
    assign position[0] = ~vUpper ? pLower[0] : pUpper[0]; 
endmodule
