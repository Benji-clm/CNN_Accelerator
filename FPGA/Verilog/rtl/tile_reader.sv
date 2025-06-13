module tile_reader #(
    parameter   BASE_ADDR   = 12'h000,
    parameter   TILE_W      = 24,
    parameter   TILE_H      = 24,
    parameter   WORD_BITS   = 256,
    parameter   PIX_BITS    = 8
)(
    input  logic               clk,
    input  logic               rst_n,

    output logic [7:0]         pixel,
    output logic               pixel_valid,
    // input  logic               pixel_ready,

    output logic [11:0]        bram_addr,
    input  logic [WORD_BITS-1:0] bram_rdata
);
    logic  [4:0] col_idx;
    logic  [4:0] row_idx;

    logic [WORD_BITS-1:0] col_buf;
    logic                 col_buf_valid;

    assign bram_addr = BASE_ADDR + col_idx;

    typedef enum logic [1:0] {FETCH, WAIT, STREAM} state_t;
    state_t state, next;

    always_comb begin
        next = state;
        case (state)
            FETCH : next = WAIT;
            WAIT  : next = STREAM;
            STREAM: begin
                if (pixel_valid && row_idx == TILE_H-1)
                    next = FETCH;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state        <= FETCH;
            col_idx      <= 0;
            row_idx      <= 0;
            col_buf_valid<= 1'b0;
        end else begin
            state <= next;

            case (state)
                FETCH: ;
                WAIT : begin
                    col_buf       <= bram_rdata;
                    col_buf_valid <= 1'b1;
                    row_idx       <= 0;
                end
                STREAM: if (pixel_valid) begin
                    if (row_idx == TILE_H-1) begin
                        row_idx  <= 0;
                        col_idx  <= (col_idx == TILE_W-1) ? 0 : col_idx + 1;
                        col_buf_valid <= 1'b0;
                    end else begin
                        row_idx  <= row_idx + 1;
                    end
                end
            endcase
        end
    end

    logic [7:0] bit_idx;
    assign bit_idx = row_idx * PIX_BITS;

    assign pixel        = col_buf[bit_idx +: PIX_BITS];
    assign pixel_valid  = (state == STREAM) && col_buf_valid;

endmodule
