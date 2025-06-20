module top_tiler #(
    parameter int   X_SIZE = 640,
    parameter int   Y_SIZE = 480,
    parameter       READ_WIDTH = 256,
    parameter       PIX_BITS    = 8
)(
    input logic                 out_stream_aclk,
    input logic                 periph_resetn,
    input logic                 ready,
    input logic                 valid_int,

    output logic                 first,
    output logic [PIX_BITS-1:0]  pixel,
    output logic                 lastx,
    output logic                 lasty,

    output logic [9:0]           x,
    output logic [8:0]           y,

    // BRAM I/O
    input  logic [READ_WIDTH-1:0] bram_rdata,
    output logic [11:0]        bram_addr
);

//================================================================
// 1. VGA Sync / Screen Position Counters
//================================================================
assign first = (x == 0) & (y == 0);
assign lastx = (x == X_SIZE-1);
assign lasty = (y == Y_SIZE-1);

always @(posedge out_stream_aclk) begin
    if (!periph_resetn) begin
        x <= 0;
        y <= 0;
    end else begin
        if (ready & valid_int) begin
            if (lastx) begin
                x <= 9'd0;
                if (lasty) y <= 9'd0;
                else y <= y + 9'd1;
            end
            else x <= x + 9'd1;
        end
    end
end

//================================================================
// 2. Tile Layout and Region Detection
//================================================================
localparam SCALE_FACTOR = 4;

// --- Tile 1-4 (From Layer 1) ---
localparam TILE_L1_W = 24;
localparam TILE_L1_DISPLAY_W = TILE_L1_W * SCALE_FACTOR;
localparam TILE_1_START_X = 20;
localparam TILE_1_START_Y = 20;
localparam TILE_2_START_X = TILE_1_START_X + TILE_L1_DISPLAY_W + 20;
localparam TILE_2_START_Y = 20;
localparam TILE_3_START_X = 20;
localparam TILE_3_START_Y = TILE_1_START_Y + TILE_L1_DISPLAY_W + 20;
localparam TILE_4_START_X = TILE_2_START_X;
localparam TILE_4_START_Y = TILE_3_START_Y;

// --- Tile 5 (From Layer 2) ---
localparam TILE_L2_W = 10;
localparam TILE_L2_DISPLAY_W = TILE_L2_W * SCALE_FACTOR;
localparam TILE_5_START_X = TILE_2_START_X + TILE_L1_DISPLAY_W + 20;
localparam TILE_5_START_Y = 20;

// --- Region Detection Wires ---
wire in_tile_1_region = (x >= TILE_1_START_X) && (x < TILE_1_START_X + TILE_L1_DISPLAY_W) &&
                        (y >= TILE_1_START_Y) && (y < TILE_1_START_Y + TILE_L1_DISPLAY_W);
wire in_tile_2_region = (x >= TILE_2_START_X) && (x < TILE_2_START_X + TILE_L1_DISPLAY_W) &&
                        (y >= TILE_2_START_Y) && (y < TILE_2_START_Y + TILE_L1_DISPLAY_W);
wire in_tile_3_region = (x >= TILE_3_START_X) && (x < TILE_3_START_X + TILE_L1_DISPLAY_W) &&
                        (y >= TILE_3_START_Y) && (y < TILE_3_START_Y + TILE_L1_DISPLAY_W);
wire in_tile_4_region = (x >= TILE_4_START_X) && (x < TILE_4_START_X + TILE_L1_DISPLAY_W) &&
                        (y >= TILE_4_START_Y) && (y < TILE_4_START_Y + TILE_L1_DISPLAY_W);
wire in_tile_5_region = (x >= TILE_5_START_X) && (x < TILE_5_START_X + TILE_L2_DISPLAY_W) &&
                        (y >= TILE_5_START_Y) && (y < TILE_5_START_Y + TILE_L2_DISPLAY_W);

//================================================================
// 3. Tile Reader Instantiation (5 Instances)
//================================================================
logic [11:0] addr_tile [4:0];
logic [PIX_BITS-1:0] pixel_tile [4:0];
logic pixel_valid_tile [4:0];
logic advance_tile [4:0];

genvar i;
generate
    for (i = 0; i < 4; i = i + 1) begin : gen_l1_readers
        tile_reader #(
            .BASE_ADDR(i * 30), // Use addresses 0, 30, 60, 90
            .TILE_W(TILE_L1_W), .TILE_H(TILE_L1_W),
            .WORD_BITS(READ_WIDTH), .PIX_BITS(PIX_BITS)
        ) tile_reader_l1_inst (
            .clk(out_stream_aclk), .rst_n(periph_resetn),
            .advance_pixel(advance_tile[i]),
            .pixel(pixel_tile[i]), .pixel_valid(pixel_valid_tile[i]),
            .bram_addr(addr_tile[i]), .bram_rdata(bram_rdata)
        );
    end
endgenerate

tile_reader #(
    .BASE_ADDR(12'h0A0), // Address for Layer 2 feature map
    .TILE_W(TILE_L2_W), .TILE_H(TILE_L2_W),
    .WORD_BITS(READ_WIDTH), .PIX_BITS(PIX_BITS)
) tile_reader_l2_inst (
    .clk(out_stream_aclk), .rst_n(periph_resetn),
    .advance_pixel(advance_tile[4]),
    .pixel(pixel_tile[4]), .pixel_valid(pixel_valid_tile[4]),
    .bram_addr(addr_tile[4]), .bram_rdata(bram_rdata)
);


//================================================================
// 4. Scaling and Advance Logic
//================================================================
logic [1:0] scale_x [4:0];
logic [1:0] scale_y [4:0];

always_ff @(posedge out_stream_aclk) begin
    if (!periph_resetn) begin
        for (int j = 0; j < 5; j++) begin
            scale_x[j] <= 0;
            scale_y[j] <= 0;
        end
    end else if (ready & valid_int) begin
        // --- Handle Scaling for each tile region ---
        if (in_tile_1_region) scale_y[0] <= (y - TILE_1_START_Y) % SCALE_FACTOR; else scale_y[0] <= 0;
        if (in_tile_1_region) scale_x[0] <= (x - TILE_1_START_X) % SCALE_FACTOR; else scale_x[0] <= 0;

        if (in_tile_2_region) scale_y[1] <= (y - TILE_2_START_Y) % SCALE_FACTOR; else scale_y[1] <= 0;
        if (in_tile_2_region) scale_x[1] <= (x - TILE_2_START_X) % SCALE_FACTOR; else scale_x[1] <= 0;

        if (in_tile_3_region) scale_y[2] <= (y - TILE_3_START_Y) % SCALE_FACTOR; else scale_y[2] <= 0;
        if (in_tile_3_region) scale_x[2] <= (x - TILE_3_START_X) % SCALE_FACTOR; else scale_x[2] <= 0;
        
        if (in_tile_4_region) scale_y[3] <= (y - TILE_4_START_Y) % SCALE_FACTOR; else scale_y[3] <= 0;
        if (in_tile_4_region) scale_x[3] <= (x - TILE_4_START_X) % SCALE_FACTOR; else scale_x[3] <= 0;

        if (in_tile_5_region) scale_y[4] <= (y - TILE_5_START_Y) % SCALE_FACTOR; else scale_y[4] <= 0;
        if (in_tile_5_region) scale_x[4] <= (x - TILE_5_START_X) % SCALE_FACTOR; else scale_x[4] <= 0;
    end
end

// Only advance the tile reader on the last pixel of a scaled group
assign advance_tile[0] = (scale_x[0] == SCALE_FACTOR-1) && (scale_y[0] == SCALE_FACTOR-1) && (ready & valid_int) && in_tile_1_region;
assign advance_tile[1] = (scale_x[1] == SCALE_FACTOR-1) && (scale_y[1] == SCALE_FACTOR-1) && (ready & valid_int) && in_tile_2_region;
assign advance_tile[2] = (scale_x[2] == SCALE_FACTOR-1) && (scale_y[2] == SCALE_FACTOR-1) && (ready & valid_int) && in_tile_3_region;
assign advance_tile[3] = (scale_x[3] == SCALE_FACTOR-1) && (scale_y[3] == SCALE_FACTOR-1) && (ready & valid_int) && in_tile_4_region;
assign advance_tile[4] = (scale_x[4] == SCALE_FACTOR-1) && (scale_y[4] == SCALE_FACTOR-1) && (ready & valid_int) && in_tile_5_region;


//================================================================
// 5. Final Pixel and BRAM Address Mux
//================================================================
always_comb begin
    // Default values
    pixel = 8'h00;  // Black background
    bram_addr = 12'h000;
    
    if (in_tile_1_region) begin
        bram_addr = addr_tile[0];
        if (pixel_valid_tile[0]) pixel = pixel_tile[0];
    end
    else if (in_tile_2_region) begin
        bram_addr = addr_tile[1];
        if (pixel_valid_tile[1]) pixel = pixel_tile[1];
    end
    else if (in_tile_3_region) begin
        bram_addr = addr_tile[2];
        if (pixel_valid_tile[2]) pixel = pixel_tile[2];
    end
    else if (in_tile_4_region) begin
        bram_addr = addr_tile[3];
        if (pixel_valid_tile[3]) pixel = pixel_tile[3];
    end
    else if (in_tile_5_region) begin
        bram_addr = addr_tile[4];
        if (pixel_valid_tile[4]) pixel = pixel_tile[4];
    end
end

endmodule