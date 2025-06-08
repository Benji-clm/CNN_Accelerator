module top_red_dec #(
    parameter DATA_WIDTH = 16,
    parameter N_MATS = 10
)(
    input  logic                            clk,
    input  logic                            rst,
    input  logic                            valid_in,
    input  logic [DATA_WIDTH-1:0]           column [N_MATS-1:0][2],

    output logic [DATA_WIDTH-1:0]           max,
    output logic [$clog2(N_MATS)-1:0]       index,
    output logic                            valid_out
);

logic valid_reduction;
logic [DATA_WIDTH-1:0] sum [N_MATS-1:0];


top_reduction #(
    .N_MATS(N_MATS),
    .DATA_WIDTH(DATA_WIDTH)
) top_red (
    .clk(clk),
    .rst(rst),
    .valid_in(valid_in),
    .column(column),
    .valid_out(valid_reduction),
    .sum(sum)
);

digit_dec #(
    .N_MATS(N_MATS),
    .DATA_WIDTH(DATA_WIDTH)
) decision (
    .clk(clk),
    .rst(rst),
    .in_sum(sum),
    .valid_in(valid_reduction),
    .max(max),
    .index(index),
    .valid_out(valid_out)
);

endmodule
