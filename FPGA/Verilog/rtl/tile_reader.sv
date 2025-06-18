module tile_reader #(
    parameter   BASE_ADDR   = 12'h000,
    parameter   TILE_W      = 24,
    parameter   TILE_H      = 24,
    parameter   WORD_BITS   = 256,
    parameter   PIX_BITS    = 8
)(
    input  logic               clk,
    input  logic               rst_n,
    input  logic               advance_pixel,  // Control signal from tiler

    output logic [7:0]         pixel,
    output logic               pixel_valid,

    output logic [11:0]        bram_addr,
    input  logic [WORD_BITS-1:0] bram_rdata
);
    // --- LOGIC RE-ARCHITECTED FOR ROW-MAJOR (RASTER) SCAN ---

    // Internal counters for the current pixel within the tile
    logic [$clog2(TILE_W)-1:0] tile_x; // Formerly row_idx, now tracks horizontal position
    logic [$clog2(TILE_H)-1:0] tile_y; // Formerly col_idx, now tracks vertical position

    // This buffer holds one entire ROW of the tile
    logic [WORD_BITS-1:0] row_buf;
    logic                 row_buf_valid;

    // The BRAM address is now determined by the ROW we want to fetch
    assign bram_addr = BASE_ADDR + tile_y;

    // --- State Machine (Largely Unchanged) ---
    typedef enum logic [1:0] {FETCH, WAIT1, WAIT2, STREAM} state_t;
    state_t state, next;

    always_comb begin
        next = state;
        case (state)
            FETCH : next = WAIT1;
            WAIT1 : next = WAIT2;
            WAIT2 : next = STREAM;
            STREAM: begin
                // A new FETCH is needed when we finish streaming a ROW.
                // This happens when tile_x is at the end of the current row.
                if (pixel_valid && advance_pixel && tile_x == TILE_W-1)
                    next = FETCH;
            end
        endcase
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            state         <= FETCH;
            tile_x        <= 0;
            tile_y        <= 0;
            row_buf_valid <= 1'b0;
        end else begin
            state <= next;

            case (state)
                FETCH: ;
                WAIT1: ;
                WAIT2: begin
                    // We've fetched a new row from BRAM, latch it.
                    row_buf       <= bram_rdata;
                    row_buf_valid <= 1'b1;
                    tile_x        <= 0; // Reset horizontal counter for the new row
                end
                STREAM: if (pixel_valid && advance_pixel) begin
                    // This is the core raster-scan logic for the tile.
                    // When told to advance, we move left-to-right, then top-to-bottom.
                    if (tile_x == TILE_W-1) begin
                        // At the end of a row, wrap tile_x and advance tile_y
                        tile_x   <= 0;
                        tile_y   <= (tile_y == TILE_H-1) ? 0 : tile_y + 1;
                        row_buf_valid <= 1'b0; // Invalidate buffer, we need a new row
                    end else begin
                        // Move to the next pixel in the current row
                        tile_x   <= tile_x + 1;
                    end
                end
            endcase
        end
    end

    // --- Pixel Selection Logic ---
    logic [$clog2(WORD_BITS)-1:0] bit_idx;
    // The pixel is selected from the row buffer using the HORIZONTAL (tile_x) position
    assign bit_idx = tile_x * PIX_BITS;
    assign pixel = row_buf[bit_idx +: PIX_BITS];
    assign pixel_valid  = (state == STREAM) && row_buf_valid;

endmodule
