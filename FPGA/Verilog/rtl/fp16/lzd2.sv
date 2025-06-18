module lzd2 (
    input  logic [1:0] a,
    output logic       position,
    output logic       valid
);
    assign valid = a[1] | a[0]; 
    assign position = ~a[1]; 
endmodule
