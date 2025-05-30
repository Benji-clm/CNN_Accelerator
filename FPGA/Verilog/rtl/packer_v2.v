module packer_v2(
    input           aclk,
    input           aresetn,

    input  [7:0]    r, g, b,
    input           eol,
    output          in_stream_ready,
    input           valid,
    input           sof, 

    output [31:0]   out_stream_tdata,
    output [3:0]    out_stream_tkeep,
    output          out_stream_tlast,
    input           out_stream_tready,
    output          out_stream_tvalid,
    output [0:0]    out_stream_tuser 
);

assign out_stream_tdata  = {8'b0, r, g, b};
assign out_stream_tkeep  = 4'b0111;
assign out_stream_tvalid = valid;
assign out_stream_tlast  = eol;
assign out_stream_tuser  = sof;
assign in_stream_ready   = out_stream_tready;

endmodule
