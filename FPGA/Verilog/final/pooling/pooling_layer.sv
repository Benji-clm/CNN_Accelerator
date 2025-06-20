module pooling_layer #(
    parameter WINDOWS     = 12
)(
    input logic clk,
    input logic rst,
    input logic valid_in,
    input logic [15:0] input_column [WINDOWS * 2 - 1:0],
    output logic valid_out,
    output logic [15:0] output_column [WINDOWS - 1:0]
);

    logic store;
    pooling_control pooling_control(
        .clk(clk),
        .rst(rst),
        .valid_in(valid_in),
        .store(store),
        .valid_out(valid_out)
    );
    // Combinational logic for 2x2 max pooling
    for (genvar j = 0; j < WINDOWS; j++) begin : pool_gen
        pooling_2x2 pool(
            .clk(clk),
            .rst(rst),
            .a(input_column[2*j]),
            .b(input_column[2*j+1]),
            .store(store), 
            .pooled_value(output_column[j])
        );
    end

endmodule
