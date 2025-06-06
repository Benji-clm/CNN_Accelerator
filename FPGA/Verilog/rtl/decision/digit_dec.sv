module digit_dec #(
    parameter int DATA_WIDTH = 16,
    parameter int N_MATS     = 10
)(
    input  logic                                 clk,
    input  logic                                 rst,

    input  logic [DATA_WIDTH-1:0]                in_sum [N_MATS],
    input  logic                                 valid_in,

    output logic [DATA_WIDTH-1:0]                max,
    output logic [$clog2(N_MATS)-1:0]            index,
    output logic                                 valid_out
);

    localparam int IDX_W = $clog2(N_MATS);

    // yoooo we can use struct in SystemVerilog,
    // makes it way easier to keep track of index whilst avoiding the pain of doubling the comparaisons
    typedef struct packed {
        logic [DATA_WIDTH-1:0] val;
        logic [IDX_W-1:0]      idx;
    } pair_t;

    localparam int PADDED  = 1 << $clog2(N_MATS);
    localparam int STAGES  = $clog2(PADDED);

    pair_t stage0 [PADDED];

    genvar i;
    generate
        for (i = 0; i < PADDED; i++) begin : g_in
            always_ff @(posedge clk) begin
                if (rst) begin
                    stage0[i].val <= '0;
                    stage0[i].idx <= '0;
                end else if (valid_in) begin
                    stage0[i].val <= (i < N_MATS) ? in_sum[i] : '0;
                    stage0[i].idx <= i[IDX_W-1:0];
                end
            end
        end
    endgenerate


    pair_t stage  [STAGES:0][PADDED];

    // We can just use generatre for the decision tree, yay (why did I write the other stupid version IDK)
    generate
        for (i = 0; i < PADDED; i++) begin
            always_comb begin
                stage[0][i] = stage0[i];
            end
        end
    endgenerate

    genvar s, k;
    generate
        for (s = 0; s < STAGES; s++) begin : g_stage
            localparam int CUR_LEN  = PADDED >> s;
            localparam int NEXT_LEN = CUR_LEN >> 1;

            for (k = 0; k < NEXT_LEN; k++) begin : g_cmp
                // FP16 comparison signals
                logic a_gt_b, a_eq_b, a_lt_b;
                
                // Instantiate FP16 comparator
                cmpfp16 fp_cmp (
                    .a(stage[s][2*k].val),
                    .b(stage[s][2*k+1].val),
                    .a_gt_b(a_gt_b),
                    .a_eq_b(a_eq_b),
                    .a_lt_b(a_lt_b)
                );
                
                always_ff @(posedge clk) begin
                    if (rst) begin
                        stage[s+1][k] <= '0;
                    end else begin
                        // Choose the larger value (a_gt_b or a_eq_b means a >= b)
                        if (a_gt_b || a_eq_b) begin
                            stage[s+1][k] <= stage[s][2*k];
                        end else begin
                            stage[s+1][k] <= stage[s][2*k+1];
                        end
                    end
                end
            end
        end
    endgenerate

    logic [STAGES:0] valid_pipe;

    always_ff @(posedge clk) begin
        if (rst)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[STAGES-1:0], valid_in};
    end

    // =========================================================================
    //  Drive outputs (ready STAGES+1 clocks after valid_in).
    // =========================================================================
    always_ff @(posedge clk) begin
        if (rst) begin
            max       <= '0;
            index     <= '0;
            valid_out <= 1'b0;
        end else begin
            max       <= stage[STAGES][0].val;
            index     <= stage[STAGES][0].idx;
            valid_out <= valid_pipe[STAGES];
        end
    end

endmodule
