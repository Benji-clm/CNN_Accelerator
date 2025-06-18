module ReLU_column #(
    parameter COLUMN_SIZE = 24
) (
    input logic  [15:0] data_in [COLUMN_SIZE],
    output logic [15:0] data_out [COLUMN_SIZE]
);

    genvar i;
    generate
        for (i = 0; i < COLUMN_SIZE; i = i + 1) begin : relu_instance
            ReLU reLU_unit (
                .data_in(data_in[i]),
                .data_out(data_out[i])
            );
        end
    endgenerate

endmodule
