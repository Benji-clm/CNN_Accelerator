// ====================================================================
// Reads from the BRAM and outputs on the HDMI (essentially a new Pixel Generator)
// - Compared to O.G. Pixel Generator - we are missing dynamic controls
// ====================================================================
module display_tiler #(
    parameter int TILE_W = 96,
    parameter int TILE_H = 96,
    parameter int COLS   = 4,
    parameter int NM     = 22      // number of maps
)(
    input  logic        pix_clk,
    // input  logic        rst,  // no need for rst?
    // BRAM
    output logic [15:0] bram_addr,
    input  logic [7:0]  bram_q,
    // VTC pixel counters (horizontal/vertical or x/y)
    input  logic [11:0] h_cnt,
    input  logic [11:0] v_cnt,
    // AXI-Stream out
    output logic [23:0] tdata,
    output logic        tvalid,
    output logic        tlast,
    output logic [3:0]  tkeep,
    output logic        tuser
);


// struct with the info of each map
typedef struct packed {
   logic [14:0] base;   // 0 … 32k-1   byte address in BRAM
   logic [5:0]  w;      // native width   (max 63)
   logic [5:0]  h;      // native height
   logic [6:0]  S;      // integer scale (1 … 64)
   logic [7:0]  min;    // for per-map contrast
   logic [15:0] gain;   // (255/(max-min))<<8
} map_cfg_t;

(* ram_style = "distributed" *)
map_cfg_t cfg [0:21];   // 22 maps

// ------------------------------------------------------------------
// Tile-configuration ROM / register file
// ------------------------------------------------------------------
initial begin
    // ─── Tile 0 : Input image ─────────────────────────────────────
    cfg[0] = '{base:15'h0000, w:6'd28, h:6'd28, S:7'd3,  min:8'd0, gain:16'd255};

    // ─── Tiles 1-4 : 24×24 Conv-1 maps (stride 0x240) ────────────
    cfg[1] = '{base:15'h0320, w:24, h:24, S:4, min:0, gain:255};
    cfg[2] = '{base:15'h0560, w:24, h:24, S:4, min:0, gain:255};
    cfg[3] = '{base:15'h07A0, w:24, h:24, S:4, min:0, gain:255};
    cfg[4] = '{base:15'h09E0, w:24, h:24, S:4, min:0, gain:255};

    // ─── Tiles 5-16 : spare 24×24 slots (same stride 0x240) ──────
    for (int i = 5; i <= 16; i++) begin
        cfg[i] = '{base: 15'h0C20 + (i-5)*15'h240,
                  w:24, h:24, S:4, min:0, gain:255};
    end

    // ─── Tiles 17-20 : 8×8 Conv-2 maps (stride 0x40) ─────────────
    cfg[17] = '{base:15'h2720, w:8, h:8, S:12, min:0, gain:255};
    cfg[18] = '{base:15'h2760, w:8, h:8, S:12, min:0, gain:255};
    cfg[19] = '{base:15'h27A0, w:8, h:8, S:12, min:0, gain:255};
    cfg[20] = '{base:15'h27E0, w:8, h:8, S:12, min:0, gain:255};

    // ─── Tile 21 : Soft-max / probabilities bar ──────────────────
    cfg[21] = '{base:15'h2820, w:10, h:1, S:9,  min:0, gain:255};

    // TODO Add another tile for the final digit choice/output
end


//------------------------------------------------------------
// derive tile indices
//------------------------------------------------------------
// for the moment simply dividing it into a 4x6 gride -> 24 tiles (we only need 22)
logic [6:0] tile_c = h_cnt / TILE_W;         // 0 … 3 -> column index
logic [6:0] tile_r = v_cnt / TILE_H;         // 0 … 5 -> row index
logic [7:0] map_id = tile_r * COLS + tile_c; // 0 … 21

logic [11:0] xt = h_cnt % TILE_W;            // 0 … 95 -> positino within tile
logic [11:0] yt = v_cnt % TILE_H;


// I know this looks awfull but gives an idea
//  | tile 0  | tile 1  | tile 2  | tile 3  |
//  -------------------------------------
//  | tile 4  | tile 5  | tile 6  | tile 7  |
//  -------------------------------------
//  | tile 8  | tile 9  | tile 10 | tile 11 |
//  -------------------------------------
//  | tile 12 | tile 13 | tile 14 | tile 15 |
//  -------------------------------------
//  | tile 16 | tile 17 | tile 18 | tile 19 |
//  -------------------------------------
//  | tile 20 | tile 21 | unused  | unused |

//------------------------------------------------------------
// fetch per-map parameters
//------------------------------------------------------------
// using the typedef/struct to find the property of each tile (hardcoded lmao)
map_cfg_t mc;

// if map_id < 22 valid tile else unused tile -> no config
always_comb mc = (map_id < NM) ? cfg[map_id] : '0;

//------------------------------------------------------------
// scale down to native coords
//------------------------------------------------------------
logic [6:0] xs = xt / mc.S;          // integer replicate ⇒ shift
logic [6:0] ys = yt / mc.S;

//------------------------------------------------------------
// check bounds
//------------------------------------------------------------
logic inside = (xt < mc.w*mc.S) && (yt < mc.h*mc.S) && (map_id < NM);

//------------------------------------------------------------
// BRAM read
//------------------------------------------------------------
assign bram_addr = mc.base + ys * mc.w + xs;

logic [7:0] pix;
always_ff @(posedge pix_clk) pix <= bram_q;   // 1-clk BRAM latency

//------------------------------------------------------------
// greyscale-to-RGB + simple contrast stretch -> essentially trippling the 8 bit grayscale value
//------------------------------------------------------------
logic [7:0] p8 = ( (pix - mc.min) * mc.gain ) >> 8;
logic [23:0] rgb = {p8,p8,p8};

//------------------------------------------------------------
// AXI-Stream protocol
//------------------------------------------------------------
assign tdata  = inside ? rgb : 24'h000000;   // black around tiles
assign tvalid = 1'b1;                        // always a pixel
assign tkeep  = 4'h7;                        // 24-bit in 32-bit lane
assign tuser  = (h_cnt==0 && v_cnt==0);      // SOF
assign tlast  = (h_cnt == (COLS*TILE_W)-1);  // EOL
endmodule

// yo I'm pretty sure this file has more comments than lines of code lmao (not my fault this AXI stream protocol thing is complicated)