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


    // BRAM I/O
    input  logic [READ_WIDTH-1:0] bram_rdata,
    output logic [11:0]        bram_addr
);

reg [9:0] x;
reg [8:0] y;

assign first = (x == 0) & (y == 0);
assign lastx = (x == X_SIZE-1);
assign lasty = (y == Y_SIZE-1);


always @(posedge out_stream_aclk) begin
    if (periph_resetn) begin
        if (ready & valid_int) begin
            if (lastx) begin
                x <= 9'd0;
                if (lasty) y <= 9'd0;
                else y <= y + 9'd1;
            end
            else x <= x + 9'd1;
        end
    end
    else begin
        x <= 0;
        y <= 0;
    end
end


localparam TILE_1_START_X = 0;
localparam TILE_1_START_Y = 0;
localparam TILE_2_START_X = 100;
localparam TILE_2_START_Y = 100;

// REGION DETECTIOOOOOOOOONNNN
wire in_tile_1_region = (x < TILE_1_START_X + TILE_1_W) &&
                        (y < TILE_1_START_Y + TILE_1_W);

wire in_tile_2_region = (x >= TILE_2_START_X) && (x < TILE_2_START_X + TILE_2_W) &&
                        (y >= TILE_2_START_Y) && (y < TILE_2_START_Y + TILE_2_W);


logic [11:0] addr_tile_1;
logic [PIX_BITS-1:0] pixel_tile_1;
logic pixel_valid_tile_1;
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
    
    .pixel(pixel_tile_1),
    .pixel_valid(pixel_valid_tile_1),
    
    .bram_addr(addr_tile_1),
    .bram_rdata(bram_rdata)
);


logic [11:0] addr_tile_2;
logic [PIX_BITS-1:0] pixel_tile_2;
logic pixel_valid_tile_2;
localparam TILE_2_W = 24;

tile_reader #(
    .BASE_ADDR(12'h00A),
    .TILE_W(TILE_2_W),
    .TILE_H(TILE_2_W),
    .WORD_BITS(READ_WIDTH),
    .PIX_BITS(PIX_BITS)
) tile_reader_inst_2 (
    .clk(out_stream_aclk),
    .rst_n(periph_resetn),
    
    // Pixel output
    .pixel(pixel_tile_2),
    .pixel_valid(pixel_valid_tile_2),
    
    // Connect to BRAM port B for reading
    .bram_addr(addr_tile_2),
    .bram_rdata(bram_rdata)
);

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
