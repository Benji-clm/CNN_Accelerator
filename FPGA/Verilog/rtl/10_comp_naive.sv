module 10_comp_naive #(
    parameter DATA_WIDTH = 16,
    parameter N_MATS = 10
)(
    input logic                             clk,
    input logic                             rst,
    input logic [DATA_WIDTH-1:0]            sum   [N_MATS-1:0],
    output logic [DATA_WIDTH-1:0]           max,
    output logic                            index,
    output logic                            valid_out
);

typedef enum logic [2:0] {STAGE_1, STAGE_2, STAGE_3, STAGE_4, OUTPUT} state_t;
state_t state, next_state;

logic [DATA_WIDTH-1:0] win_sum1_stage1, win_sum2_stage1, win_sum3_stage1, win_sum4_stage1, win_sum5_stage1;
logic [DATA_WIDTH-1:0] win_sum1_stage2, win_sum2_stage2;
logic [DATA_WIDTH-1:0] win_sum1_stage3;
logic [DATA_WIDTH-1:0] win_sum1_stage4;

always_ff @(posedge clk) begin
    if (rst) begin
        state <= STAGE_1;
        valid_out <= 0;
        sum <= 0;
        index <= 0;
        out_ready <= 0;
    end else begin
        state <= next_state;
    end
end

always_comb begin 
    case (state)
        STAGE_1: begin
            win_sum1_stage1 = (sum[0] > sum[1]) ? sum[0] : sum[1];
            win_sum2_stage1 = (sum[2] > sum[3]) ? sum[2] : sum[3];
            win_sum3_stage1 = (sum[4] > sum[5]) ? sum[4] : sum[5];
            win_sum4_stage1 = (sum[6] > sum[7]) ? sum[6] : sum[7];
            win_sum5_stage1 = (sum[8] > sum[9]) ? sum[8] : sum[9];
            next_state = STAGE_2;
        end

        STAGE_2: begin
            win_sum1_stage2 = (win_sum1_stage1 > win_sum2_stage1) ? win_sum1_stage1 : win_sum2_stage1;
            win_sum2_stage2 = (win_sum3_stage1 > win_sum4_stage1) ? win_sum3_stage1 : win_sum4_stage1;
            next_state = STAGE_3;
        end

        STAGE_3: begin
            win_sum1_stage3 = (win_sum1_stage2 > win_sum2_stage2) ? win_sum1_stage2 : win_sum2_stage2;
            next_state = STAGE_4;
        end

        STAGE_4: begin
            win_sum1_stage4 = (win_sum1_stage3 > win_sum5_stage1) ? win_sum1_stage3 : win_sum5_stage1;;
            max = win_sum1_stage4; // Assign the max value to output
            index = 0; // Reset index for now, can be modified to track the index of max
            next_state = OUTPUT;
        end

        OUTPUT: begin
            valid_out = 1; // Signal that the output is valid
        end

        default: begin
            valid_out = 0; // Default case to reset valid_out
        end
    endcase

end

endmodule
