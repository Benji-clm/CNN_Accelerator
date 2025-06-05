module top_reduction_array #(
    parameter int N_MATS   = 10,   // number of matrices to reduce in parallel
    parameter int DATA_WIDTH   = 16    // element width
)(
    input  logic                        clk,
    input  logic                        rst,

    input  logic                        valid_in,

    // Two 16-bit rows per column, N_MATS matrices wide
    //     column[i][row]   with  i = matrix index, row = 0|1
    input  logic [DATA_WIDTH-1:0]           column [N_MATS-1:0][2],

    // Parallel results
    output logic [N_MATS-1:0]               valid_out,
    output logic [DATA_WIDTH-1:0]           sum       [N_MATS-1:0]
);

    /*------------------------------------------------------------------
     *  Instantiate N_MATS reduction units with a generate-for loop
     *----------------------------------------------------------------*/
    genvar i;
    generate
        for (i = 0; i < N_MATS; i++) begin : g_reducer

            reduction #(
                .DATA_WIDTH (DATA_WIDTH),
                .mat_height (2)          // your reduction parameter
            ) u_red (
                .clk       (clk),
                .rst       (rst),
                .valid_in  (valid_in),
                .column    (column[i]),  // 2-element column bus for matrix i
                .valid_out (valid_out[i]),
                .sum       (sum[i])
            );

        end
    endgenerate

endmodule
