// ====================================================================
// This is a modified version of the display tiler which containts the AXI registers/dynamic controls 
// to allow for run time parametrization
// 
// ====================================================================
module display_tiler #(
    parameter int TILE_W = 96,
    parameter int TILE_H = 96,
    parameter int COLS   = 4,
    parameter int NM     = 22,     // number of maps
    parameter int REG_FILE_SIZE = 64,  // 64 registers for dynamic control
    parameter int AXI_LITE_ADDR_WIDTH = 8
)(
    input  logic        pix_clk,
    input  logic        axi_resetn,     // AXI reset
    input  logic        periph_resetn,  // Peripheral reset
    
    // BRAM-B
    output logic [15:0] bram_addr,
    input  logic [7:0]  bram_q,
    
    // VTC pixel counters
    input  logic [11:0] h_cnt,
    input  logic [11:0] v_cnt,
    
    // AXI-Stream out
    output logic [23:0] tdata,
    output logic        tvalid,
    output logic        tlast,
    output logic [3:0]  tkeep,
    output logic        tuser,
    
    // AXI-Lite interface for dynamic control (optional - can be commented out)
    input  logic        s_axi_lite_aclk,
    input  logic [AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_araddr,
    output logic        s_axi_lite_arready,
    input  logic        s_axi_lite_arvalid,
    input  logic [AXI_LITE_ADDR_WIDTH-1:0] s_axi_lite_awaddr,
    output logic        s_axi_lite_awready,
    input  logic        s_axi_lite_awvalid,
    input  logic        s_axi_lite_bready,
    output logic [1:0]  s_axi_lite_bresp,
    output logic        s_axi_lite_bvalid,
    output logic [31:0] s_axi_lite_rdata,
    input  logic        s_axi_lite_rready,
    output logic [1:0]  s_axi_lite_rresp,
    output logic        s_axi_lite_rvalid,
    input  logic [31:0] s_axi_lite_wdata,
    output logic        s_axi_lite_wready,
    input  logic        s_axi_lite_wvalid
);



typedef struct packed {
   logic [14:0] base;   // 0 … 32k-1   byte address in BRAM
   logic [5:0]  w;      // native width   (max 63)
   logic [5:0]  h;      // native height
   logic [6:0]  S;      // integer scale (1 … 64)
   logic [7:0]  min;    // for per-map contrast (optional)
   logic [15:0] gain;   // (255/(max-min))<<8
} map_cfg_t;

// AXI-Lite state machine parameters (from pixel_generator)
localparam REG_FILE_AWIDTH = $clog2(REG_FILE_SIZE);
localparam AWAIT_WADD_AND_DATA = 3'b000;
localparam AWAIT_WDATA = 3'b001;
localparam AWAIT_WADD = 3'b010;
localparam AWAIT_WRITE = 3'b100;
localparam AWAIT_RESP = 3'b101;
localparam AWAIT_RADD = 2'b00;
localparam AWAIT_FETCH = 2'b01;
localparam AWAIT_READ = 2'b10;
localparam AXI_OK = 2'b00;
localparam AXI_ERR = 2'b10;

// Register file for dynamic control (can be commented out if not needed)
logic [31:0] regfile [REG_FILE_SIZE-1:0];
logic [REG_FILE_AWIDTH-1:0] writeAddr, readAddr;
logic [31:0] readData, writeData;
logic [1:0] readState;
logic [2:0] writeState;

// Dynamic control enable (set to 0 to disable AXI-Lite, 1 to enable)
localparam ENABLE_DYNAMIC_CONTROL = 1;

(* ram_style = "distributed" *)
map_cfg_t cfg [0:21];   // 22 maps


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
// AXI-Lite Register Interface (Optional Dynamic Control)
// Comment out this entire section if dynamic control not needed
//------------------------------------------------------------
generate
if (ENABLE_DYNAMIC_CONTROL) begin : gen_axi_lite

    // Initialize state machines
    initial begin
        readState = AWAIT_RADD;
        writeState = AWAIT_WADD_AND_DATA;
    end

    // AXI-Lite Read State Machine
    always_ff @(posedge s_axi_lite_aclk) begin
        readData <= regfile[readAddr];
        
        if (!axi_resetn) begin
            readState <= AWAIT_RADD;
        end
        else case (readState)
            AWAIT_RADD: begin
                if (s_axi_lite_arvalid) begin
                    readAddr <= s_axi_lite_araddr[2+:REG_FILE_AWIDTH];
                    readState <= AWAIT_FETCH;
                end
            end
            AWAIT_FETCH: begin
                readState <= AWAIT_READ;
            end
            AWAIT_READ: begin
                if (s_axi_lite_rready) begin
                    readState <= AWAIT_RADD;
                end
            end
            default: begin
                readState <= AWAIT_RADD;
            end
        endcase
    end

    // AXI-Lite Write State Machine
    always_ff @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) begin
            writeState <= AWAIT_WADD_AND_DATA;
        end
        else case (writeState)
            AWAIT_WADD_AND_DATA: begin
                case ({s_axi_lite_awvalid, s_axi_lite_wvalid})
                    2'b10: begin
                        writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                        writeState <= AWAIT_WDATA;
                    end
                    2'b01: begin
                        writeData <= s_axi_lite_wdata;
                        writeState <= AWAIT_WADD;
                    end
                    2'b11: begin
                        writeData <= s_axi_lite_wdata;
                        writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                        writeState <= AWAIT_WRITE;
                    end
                    default: begin
                        writeState <= AWAIT_WADD_AND_DATA;
                    end
                endcase
            end
            AWAIT_WDATA: begin
                if (s_axi_lite_wvalid) begin
                    writeData <= s_axi_lite_wdata;
                    writeState <= AWAIT_WRITE;
                end
            end
            AWAIT_WADD: begin
                if (s_axi_lite_awvalid) begin
                    writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                    writeState <= AWAIT_WRITE;
                end
            end
            AWAIT_WRITE: begin
                regfile[writeAddr] <= writeData;
                writeState <= AWAIT_RESP;
            end
            AWAIT_RESP: begin
                if (s_axi_lite_bready) begin
                    writeState <= AWAIT_WADD_AND_DATA;
                end
            end
            default: begin
                writeState <= AWAIT_WADD_AND_DATA;
            end
        endcase
    end

    // AXI-Lite signal assignments
    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid = (readState == AWAIT_READ);
    assign s_axi_lite_rdata = readData;
    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

end else begin : gen_no_axi_lite
    // Tie off AXI-Lite signals when disabled
    assign s_axi_lite_arready = 1'b0;
    assign s_axi_lite_rresp = AXI_ERR;
    assign s_axi_lite_rvalid = 1'b0;
    assign s_axi_lite_rdata = 32'h0;
    assign s_axi_lite_awready = 1'b0;
    assign s_axi_lite_wready = 1'b0;
    assign s_axi_lite_bvalid = 1'b0;
    assign s_axi_lite_bresp = AXI_ERR;
end
endgenerate

//------------------------------------------------------------
// Dynamic configuration override (Optional)
// Register map:
// 0x00: Global control (bit 0: enable dynamic cfg override)
// 0x04: Target tile ID for configuration
// 0x08: Base address for target tile
// 0x0C: Width and height (bits [5:0] = width, bits [13:8] = height)
// 0x10: Scale factor (bits [6:0])
// 0x14: Min value for contrast (bits [7:0])
// 0x18: Gain value for contrast (bits [15:0])
//------------------------------------------------------------
generate
if (ENABLE_DYNAMIC_CONTROL) begin : gen_dynamic_cfg
    logic use_dynamic_cfg;
    logic [4:0] target_tile_id;
    map_cfg_t dynamic_cfg;
    
    assign use_dynamic_cfg = regfile[0][0];
    assign target_tile_id = regfile[1][4:0];
    assign dynamic_cfg.base = regfile[2][14:0];
    assign dynamic_cfg.w = regfile[3][5:0];
    assign dynamic_cfg.h = regfile[3][13:8];
    assign dynamic_cfg.S = regfile[4][6:0];
    assign dynamic_cfg.min = regfile[5][7:0];
    assign dynamic_cfg.gain = regfile[6][15:0];
    
    // Override configuration for target tile if dynamic control enabled
    map_cfg_t mc_final;
    always_comb begin
        if (use_dynamic_cfg && (map_id == target_tile_id) && (map_id < NM)) begin
            mc_final = dynamic_cfg;
        end else begin
            mc_final = mc;
        end
    end
    
end else begin : gen_static_cfg
    map_cfg_t mc_final;
    assign mc_final = mc;
end
endgenerate

//------------------------------------------------------------
// scale down to native coords
//------------------------------------------------------------
logic [6:0] xs = xt / mc_final.S;          // integer replicate ⇒ shift
logic [6:0] ys = yt / mc_final.S;

//------------------------------------------------------------
// check bounds
//------------------------------------------------------------
logic inside = (xt < mc_final.w*mc_final.S) && (yt < mc_final.h*mc_final.S) && (map_id < NM);

//------------------------------------------------------------
// BRAM read
//------------------------------------------------------------
assign bram_addr = mc_final.base + ys * mc_final.w + xs;

logic [7:0] pix;
always_ff @(posedge pix_clk) pix <= bram_q;   // 1-clk BRAM latency

//------------------------------------------------------------
// greyscale-to-RGB + simple contrast stretch -> essentially trippling the 8 bit grayscale value
//------------------------------------------------------------
logic [7:0] p8 = ( (pix - mc_final.min) * mc_final.gain ) >> 8;
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
