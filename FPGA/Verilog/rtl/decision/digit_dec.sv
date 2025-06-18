/**
 * @file digit_dec.sv
 * @author Benji-clm
 * @brief Finds the maximum value and its index from a set of inputs using a pipelined tournament-style comparator tree.
 * @version 1.2
 * @date 2025-06-18
 *
 * @copyright Copyright (c) 2025
 *
 * @details
 * Revision 1.1 by Copilot: Corrected a structural issue for synthesis by separating
 * the combinational decision logic from the sequential registering logic.
 * Revision 1.2 by Copilot: Fixed a genvar scope issue by restoring the outer
 * generate loop and moving localparam declarations to the correct scope.
 */
module digit_dec #(
    parameter int DATA_WIDTH = 16,
    parameter int N_MATS     = 10
)(
    input  logic                                 clk,
    input  logic                                 rst,

    input  logic [DATA_WIDTH-1:0]                in_sum [N_MATS-1:0],
    input  logic                                 valid_in,

    output logic [DATA_WIDTH-1:0]                max,
    output logic [$clog2(N_MATS)-1:0]            index,
    output logic                                 valid_out
);

    localparam int IDX_W = $clog2(N_MATS);

    // Struct to pair a value with its original index
    typedef struct packed {
        logic [DATA_WIDTH-1:0] val;
        logic [IDX_W-1:0]      idx;
    } pair_t;

    // Parameters for the comparator tree structure
    localparam int PADDED  = 1 << $clog2(N_MATS);
    localparam int STAGES  = $clog2(PADDED);
    localparam logic [DATA_WIDTH-1:0] NEG_INF = 16'hFC00; // Define -Inf for FP16

    //================================================================
    // Stage 0: Latch and Pad Inputs
    //================================================================
    pair_t stage0 [PADDED-1:0];

    genvar i;
    generate
        for (i = 0; i < PADDED; i++) begin : g_in
            always_ff @(posedge clk) begin
                if (rst) begin
                    stage0[i].val <= '0;
                    stage0[i].idx <= '0;
                end else if (valid_in) begin
                    // Pad with negative infinity if the input doesn't exist
                    stage0[i].val <= (i < N_MATS) ? in_sum[i] : NEG_INF;
                    stage0[i].idx <= i[IDX_W-1:0];
                end
            end
        end
    endgenerate

    //================================================================
    // Pipelined Comparator Tree
    //================================================================
    pair_t stage  [STAGES:0][PADDED-1:0];

    // Connect Stage 0 latches to the input of the comparator tree
    generate
        for (i = 0; i < PADDED; i++) begin : g_stage0_connect
            always_comb begin
                stage[0][i] = stage0[i];
            end
        end
    endgenerate

    // --- THIS IS THE FULLY CORRECTED GENERATE BLOCK ---
    genvar s, k;
    generate
        // Outer loop: Iterates through the STAGES of the pipeline (s = 0, 1, 2...)
        for (s = 0; s < STAGES; s++) begin : g_stage
            
            // Localparams are now correctly INSIDE the loop, where 's' is legally defined
            localparam int CUR_LEN  = PADDED >> s;
            localparam int NEXT_LEN = CUR_LEN >> 1;

            // Inner loop: Creates the comparators for the current stage
            for (k = 0; k < NEXT_LEN; k++) begin : g_cmp
                
                logic a_gt_b, a_eq_b, a_lt_b;
                
                // 1. Instantiate the comparator
                cmpfp16 fp_cmp (
                    .a(stage[s][2*k].val),
                    .b(stage[s][2*k+1].val),
                    .a_gt_b(a_gt_b),
                    .a_eq_b(a_eq_b),
                    .a_lt_b(a_lt_b)
                );
                
                // 2. Purely combinational logic to decide the winner
                pair_t winner;
                always_comb begin
                    if (a_gt_b || a_eq_b) begin // a >= b
                        winner = stage[s][2*k];
                    end else begin // a < b
                        winner = stage[s][2*k+1];
                    end
                end

                // 3. Purely sequential logic to register the winner for the next stage
                always_ff @(posedge clk) begin
                    if (rst) begin
                        stage[s+1][k] <= '0;
                    end else begin
                        stage[s+1][k] <= winner;
                    end
                end
            end
        end
    endgenerate

    //================================================================
    // Output Logic
    //================================================================
    logic [STAGES:0] valid_pipe;

    // A simple shift register to pipeline the valid signal
    always_ff @(posedge clk) begin
        if (rst)
            valid_pipe <= '0;
        else
            valid_pipe <= {valid_pipe[STAGES-1:0], valid_in};
    end

    // Drive outputs (they will be valid STAGES+1 clocks after valid_in)
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