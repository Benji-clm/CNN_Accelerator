module top_reduction_array #(
    parameter int N_MATS   = 10,   // number of matrices to reduce in parallel
    parameter int DATA_WIDTH   = 16
)(
    input  logic                        clk,
    input  logic                        rst,

    input  logic                        valid_in,

    // N matrices, each with 2 elements, 16 bits data width
    input  logic [DATA_WIDTH-1:0]           column [N_MATS-1:0][2],

    // Parallel results
    output logic [N_MATS-1:0]               valid_out,
    output logic [DATA_WIDTH-1:0]           sum       [N_MATS-1:0]
);

    genvar i;
    generate
        for (i = 0; i < N_MATS; i++) begin : g_reducer

            reduction #(
                .DATA_WIDTH (DATA_WIDTH),
                .mat_height (2)
            ) u_red (
                .clk       (clk),
                .rst       (rst),
                .valid_in  (valid_in),
                .column    (column[i]),
                .valid_out (valid_out[i]),
                .sum       (sum[i])
            );

        end
    endgenerate

endmodule
