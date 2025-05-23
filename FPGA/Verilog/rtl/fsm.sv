module fsm #(
    parameter ADDR_W = 32
)(
    input  wire               clk,
    input  wire               rst,        // synchronous, active‑high

    input  wire               cfg_start,  // pulse from CPU
    input  logic [ADDR_W-1:0] n_layer,

    // DMA handshake
    input  wire               dma_done,  // DMA raises when image in BRAM
    output reg                dma_go,     // strobe: 1 clk to start DMA

    // MAC handshake
    input  wire               conv_done,  // array raises when frame done
    output reg                conv_start, // strobe: 1 clk to start array

    // status to CPU
    output reg                irq_done,   // 1‑cycle pulse

    output wire [ADDR_W-1:0]  rd_addr,
    output wire [ADDR_W-1:0]  wr_addr,
    output logic [ADDR_W-1:0] n_o_layer,
    output wire               valid_in,
    output wire               sop,
    output wire               eop
);

typedef enum logic [2:0] {IDLE, LOAD_KERNEL, LOAD_IMAGE, RUN, DONE} state_t;
state_t  state_reg, state_nxt;

// one‑cycle flag generator
logic first_cycle;

assign n_o_layer = n_layer;
assign rd_addr  = '0;
assign wr_addr  = '0;
assign valid_in = (state_reg == RUN);
assign sop      = (state_reg == RUN)  && first_cycle;
assign eop      = (state_reg == DONE) && first_cycle;



always_comb begin
    state_nxt  = state_reg;

    case (state_reg)
        IDLE: begin
            if (cfg_start)      state_nxt = LOAD_KERNEL;
        end
        LOAD_KERNEL: begin
            if (dma_done)      state_nxt = LOAD_IMAGE;
        end
        LOAD_IMAGE: begin
            if (dma_done)      state_nxt = RUN;
        end
        RUN: begin
            if (conv_done)      state_nxt = DONE;       // computation done
        end
        DONE: begin
            if (!cfg_start)     state_nxt = IDLE;       // wait for SW clear
        end
    endcase
end


always_ff @(posedge clk) begin
    if (rst) begin
        state_reg   <= IDLE;
        dma_go      <= 1'b0;
        conv_start  <= 1'b0;
        irq_done    <= 1'b0;
        first_cycle <= 1'b0;
    end else begin
        state_reg <= state_nxt;

        dma_go      <= 1'b0;
        conv_start  <= 1'b0;
        irq_done    <= 1'b0;

        if (first_cycle)
            first_cycle <= 1'b0;

        case (state_reg)
            IDLE: begin
                if (cfg_start) begin
                    dma_go      <= 1'b1;
                end
            end

            LOAD_KERNEL: begin
                if (dma_done) begin
                    conv_start  <= 1'b0;
                    dma_go      <= 1'b1;
                    first_cycle <= 1'b1;
                end
            end

            LOAD_IMAGE: begin
                if (dma_done) begin
                    conv_start  <= 1'b1;
                    first_cycle <= 1'b1;
                end
            end

            RUN: begin
                if (conv_done) begin
                    irq_done    <= 1'b1;
                    first_cycle <= 1'b1;
                end
            end

            DONE: begin end // do nothing, it's done.
        endcase
    end
end

endmodule
