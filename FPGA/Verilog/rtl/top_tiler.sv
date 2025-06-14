//================================================= 
//
// Purpose of this module: Keeps track of x and y position over monitor, and outputs value by reading from the BRAM independently of other things
//      Partitions the monitor -> will say "ok x and y are at these position, turn on this one tile and uses its output data"
// 
//=================================================
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

// reg [9:0] x;
// reg [8:0] y;

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


localparam TILE_1_START_X = 0;
localparam TILE_1_START_Y = 0;
localparam TILE_2_START_X = 0;
localparam TILE_2_START_Y = 100;
localparam SCALE_FACTOR = 2;

// REGION DETECTIOOOOOOOOONNNN
wire in_tile_1_region = (x < TILE_1_START_X + (TILE_1_W * SCALE_FACTOR)) &&
                        (y < TILE_1_START_Y + (TILE_1_W * SCALE_FACTOR));

wire in_tile_2_region = (x >= TILE_2_START_X) && (x < TILE_2_START_X + (TILE_2_W * SCALE_FACTOR)) &&
                        (y >= TILE_2_START_Y) && (y < TILE_2_START_Y + (TILE_2_W * SCALE_FACTOR));


logic [11:0] addr_tile_1;
logic [PIX_BITS-1:0] pixel_tile_1;
logic pixel_valid_tile_1;
logic advance_tile_1;
localparam TILE_1_W = 24;

tile_reader #(
    .BASE_ADDR(12'h000),
    .TILE_W(TILE_1_W),
    .TILE_H(TILE_1_W),
    .WORD_BITS(READ_WIDTH),
    .PIX_BITS(PIX_BITS)
) tile_reader_inst_1 (
    .clk(out_stream_aclk),
    .rst_n(periph_resetn),
    .advance_pixel(advance_tile_1),
    
    .pixel(pixel_tile_1),
    .pixel_valid(pixel_valid_tile_1),
    
    .bram_addr(addr_tile_1),
    .bram_rdata(bram_rdata)
);


logic [11:0] addr_tile_2;
logic [PIX_BITS-1:0] pixel_tile_2;
logic pixel_valid_tile_2;
logic advance_tile_2;
localparam TILE_2_W = 24;

tile_reader #(
    .BASE_ADDR(12'h000),
    .TILE_W(TILE_2_W),
    .TILE_H(TILE_2_W),
    .WORD_BITS(READ_WIDTH),
    .PIX_BITS(PIX_BITS)
) tile_reader_inst_2 (
    .clk(out_stream_aclk),
    .rst_n(periph_resetn),
    .advance_pixel(advance_tile_2),
    
    // Pixel output
    .pixel(pixel_tile_2),
    .pixel_valid(pixel_valid_tile_2),
    
    // Connect to BRAM port B for reading
    .bram_addr(addr_tile_2),
    .bram_rdata(bram_rdata)
);

// Scaling control signals
logic [1:0] tile_1_scale_x, tile_1_scale_y;
logic [1:0] tile_2_scale_x, tile_2_scale_y;

// Scaling logic
always_ff @(posedge out_stream_aclk) begin
    if (!periph_resetn) begin
        tile_1_scale_x <= 0;
        tile_1_scale_y <= 0;
        tile_2_scale_x <= 0;
        tile_2_scale_y <= 0;
    end else if (ready & valid_int) begin
        // Handle tile 1 scaling
        if (in_tile_1_region) begin
            // Get relative position within tile region
            logic [9:0] rel_x = x - TILE_1_START_X;
            logic [8:0] rel_y = y - TILE_1_START_Y;
            
            tile_1_scale_x <= rel_x % SCALE_FACTOR;
            tile_1_scale_y <= rel_y % SCALE_FACTOR;
        end else begin
            tile_1_scale_x <= 0;
            tile_1_scale_y <= 0;
        end
        
        // Handle tile 2 scaling
        if (in_tile_2_region) begin
            // Get relative position within tile region
            logic [9:0] rel_x = x - TILE_2_START_X;
            logic [8:0] rel_y = y - TILE_2_START_Y;
            
            tile_2_scale_x <= rel_x % SCALE_FACTOR;
            tile_2_scale_y <= rel_y % SCALE_FACTOR;
        end else begin
            tile_2_scale_x <= 0;
            tile_2_scale_y <= 0;
        end
    end
end

// Only advance tile reader when we've finished scaling current pixel
assign advance_tile_1 = (tile_1_scale_x == SCALE_FACTOR-1) && (tile_1_scale_y == SCALE_FACTOR-1) && 
                        (ready & valid_int) && in_tile_1_region;
                        
assign advance_tile_2 = (tile_2_scale_x == SCALE_FACTOR-1) && (tile_2_scale_y == SCALE_FACTOR-1) && 
                        (ready & valid_int) && in_tile_2_region;

always_comb begin 
    // Default values
    pixel = 8'h00;  // Black by default (nice separation between feature maps)
    bram_addr = 12'h000;
    
    if (in_tile_1_region) begin
        bram_addr = addr_tile_1;
        if (pixel_valid_tile_1) begin
            pixel = pixel_tile_1;
        end
    end
    else if (in_tile_2_region) begin
        bram_addr = addr_tile_2;
        if (pixel_valid_tile_2) begin
            pixel = pixel_tile_2;
        end
    end
end

endmodule
