// ====================================================================
//  Feature-map capture with back-pressure
//  – Latches one full column (PIX_H×24-bit) in a single clock
//  – Writes PIX_H greyscale bytes to the shared display-BRAM
//    at 1-byte/clk
//  – De-asserts ready_col while writing, so the CNN waits
// ====================================================================
module fmap_capture #(
    parameter int PIX_W     = 24,      // map width  (columns)
    parameter int PIX_H     = 24,      // map height (rows)
    parameter int BASE_ADDR = 0,       // byte offset in display BRAM
    parameter int PIX_BITS  = 8        // we store 8-bit greyscale
)(
    input  logic                      clk,
    input  logic                      rst_n,

    // ---------- CNN side (ready/valid handshake) ----------
    input  logic                      valid_col,               // new column available
    output logic                      ready_col,               // writer can accept - used to stall the CNN whilst we write
    input  logic [23:0]               data_col [PIX_H-1:0],    // PIX_H rows in parallel

    // ---------- display-BRAM write port (port-A) ----------
    output logic [15:0]               bram_addr,
    output logic [PIX_BITS-1:0]       bram_wdata,
    output logic                      bram_we,

    // ---------- completion pulse --------------
    output logic                      done                     // 1-clk when full map stored (should be used by next layer so that there is no two modules using the writing tap simulatenoulsy)
);

    // ----------------------------------------------------------------
    // Internal state
    // ----------------------------------------------------------------
    localparam int ROWS = PIX_H;
    localparam int COLS = PIX_W;

    logic [7:0] col_buf [0:ROWS-1];                    // one-column buffer
    logic [$clog2(COLS)-1:0]  col_ptr;                 // which column we’re writing
    logic [$clog2(ROWS)-1:0]  row_ptr;                 // which row inside that column

    typedef enum logic {IDLE, WRITE_ROWS} st_t;
    st_t st;

    // ----------------------------------------------------------------
    // Main FSM
    // ----------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            st        <= IDLE;
            col_ptr   <= '0;
            row_ptr   <= '0;
            bram_we   <= 1'b0;
            ready_col <= 1'b1;   // ready after reset
            done      <= 1'b0;
        end
        else begin
            // defaults every cycle
            bram_we   <= 1'b0;
            done      <= 1'b0;

            unique case (st)
            // ================================================= IDLE
            IDLE: begin
                ready_col <= 1'b1;                 // tell CNN we can take a column
                if (valid_col && ready_col) begin
                    // latch the entire column in one clock
                    for (int r = 0; r < ROWS; r++)
                        col_buf[r] <= data_col[r][23:16];     // MS byte → greyscale
                    row_ptr   <= 0;
                    ready_col <= 1'b0;            // stall CNN while we write it out
                    st        <= WRITE_ROWS;
                end
            end

            // ========================================= WRITE each row
            WRITE_ROWS: begin
                bram_we    <= 1'b1;
                bram_addr  <= BASE_ADDR + row_ptr*COLS + col_ptr;
                bram_wdata <= col_buf[row_ptr];

                if (row_ptr == ROWS-1) begin          // last row of this column
                    row_ptr <= 0;
                    if (col_ptr == COLS-1) begin      // whole map finished
                        col_ptr   <= 0;
                        done      <= 1'b1;
                        st        <= IDLE;            // accept next map
                    end
                    else begin                        // prepare for next column
                        col_ptr   <= col_ptr + 1;
                        st        <= IDLE;            // re-enter IDLE → ready_col=1
                    end
                end
                else begin
                    row_ptr <= row_ptr + 1;           // next row in same column
                end
            end
            endcase
        end
    end
endmodule
