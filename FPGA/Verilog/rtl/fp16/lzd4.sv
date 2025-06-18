module lzd4 (
    input  logic [3:0] a,
    output logic [1:0] position,
    output logic       valid
);
    wire pUpper, pLower, vUpper, vLower;

    lzd2 lzd2_1 (.a(a[3:2]), .position(pUpper), .valid(vUpper)); 
    lzd2 lzd2_2 (.a(a[1:0]), .position(pLower), .valid(vLower)); 

    assign valid = vUpper | vLower; 
    assign position[1] = ~vUpper; 
    assign position[0] = ~vUpper ? pLower : pUpper; 
endmodule
