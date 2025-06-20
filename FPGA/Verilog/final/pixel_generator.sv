module pixel_generator(
    input           out_stream_aclk,
    input           s_axi_lite_aclk,
    input           axi_resetn,
    input           periph_resetn,

    // AXI-Stream Video Output
    output [31:0]   out_stream_tdata,
    output [3:0]    out_stream_tkeep,
    output          out_stream_tlast,
    input           out_stream_tready,
    output          out_stream_tvalid,
    output [0:0]    out_stream_tuser, 

    // AXI-Lite Control Interface
    input [AXI_LITE_ADDR_WIDTH-1:0]     s_axi_lite_araddr,
    output          s_axi_lite_arready,
    input           s_axi_lite_arvalid,
    input [AXI_LITE_ADDR_WIDTH-1:0]     s_axi_lite_awaddr,
    output          s_axi_lite_awready,
    input           s_axi_lite_awvalid,
    input           s_axi_lite_bready,
    output [1:0]    s_axi_lite_bresp,
    output          s_axi_lite_bvalid,
    output [31:0]   s_axi_lite_rdata,
    input           s_axi_lite_rready,
    output [1:0]    s_axi_lite_rresp,
    output          s_axi_lite_rvalid,
    input  [31:0]   s_axi_lite_wdata,
    output          s_axi_lite_wready,
    input           s_axi_lite_wvalid,

    input [255:0]			bram_rddata_a,
    output [11:0]			bram_addr_a, // address
    output 					bram_clk_a, // clock
    output [255:0]			bram_wrdata_a, // written DATA into BRAM
    output					bram_en_a, // global enable
    output 					bram_rst_a, // reset	
    output [31:0]      		bram_we_a, // write enable for each byte

    // BRAM SIGNALLLLSSSSS - port B
    input [255:0]			bram_rddata_b,
    output [11:0]			bram_addr_b, // address
    output 					bram_clk_b, // clock
    output [255:0]			bram_wrdata_b, // written DATA into BRAM
    output					bram_en_b, // global enable
    output 					bram_rst_b, // reset	
    output [31:0]      	    bram_we_b, // write enable for each byte


    // BRAM Processing System
    input [255:0]			bram_rddata_a_ps,
    output [11:0]			bram_addr_a_ps, // address
    output 					bram_clk_a_ps, // clock
    output [255:0]			bram_wrdata_a_ps, // written DATA into BRAM
    output					bram_en_a_ps, // global enable
    output 					bram_rst_a_ps, // reset	
    output [31:0]     	    bram_we_a_ps // write enable for each byte
    );

    localparam X_SIZE = 640;
    localparam Y_SIZE = 480;
    parameter  REG_FILE_SIZE = 8;
    localparam REG_FILE_AWIDTH = $clog2(REG_FILE_SIZE);
    parameter  AXI_LITE_ADDR_WIDTH = 8;

    // AXI Lite FSM States and Responses
    localparam AWAIT_WADD_AND_DATA = 3'b000, AWAIT_WDATA = 3'b001, AWAIT_WADD = 3'b010, AWAIT_WRITE = 3'b100, AWAIT_RESP = 3'b101;
    localparam AWAIT_RADD = 2'b00, AWAIT_FETCH = 2'b01, AWAIT_READ = 2'b10;
    localparam AXI_OK = 2'b00, AXI_ERR = 2'b10;

    reg [31:0]                          regfile [REG_FILE_SIZE-1:0];
    reg [REG_FILE_AWIDTH-1:0]           writeAddr, readAddr;
    reg [31:0]                          readData, writeData;
    reg [1:0]                           readState = AWAIT_RADD;
    reg [2:0]                           writeState = AWAIT_WADD_AND_DATA;

    //================================================================
    // ADDED: CDC logic for max and index signals
    //================================================================
    logic [15:0] max_pclk;
    logic [$clog2(10)-1:0] index_pclk;
    logic [15:0] max_sync1, max_sync2;
    logic [$clog2(10)-1:0] index_sync1, index_sync2;

    // First stage registers in peripheral clock domain to avoid metastability
    always_ff @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            max_pclk <= '0;
            index_pclk <= '0;
        end else begin
            max_pclk <= max;
            index_pclk <= index;
        end
    end

    // Two-flop synchronizer to cross from out_stream_aclk to s_axi_lite_aclk
    always_ff @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) begin
            max_sync1 <= '0;
            max_sync2 <= '0;
            index_sync1 <= '0;
            index_sync2 <= '0;
        end else begin
            max_sync1 <= max_pclk;
            max_sync2 <= max_sync1;
            index_sync1 <= index_pclk;
            index_sync2 <= index_sync1;
        end
    end

    // AXI-Lite Read Logic (Unchanged and Correct)
    always @(posedge s_axi_lite_aclk) begin
        case (readAddr)
            3'd2:    readData <= {16'b0, max_sync2};   // If address is 2, provide max
            3'd3:    readData <= {28'b0, index_sync2}; // If address is 3, provide index
            default: readData <= regfile[readAddr];    // For all other addresses, provide the regfile value
        endcase
        if (!axi_resetn) readState <= AWAIT_RADD;
        else case (readState)
            AWAIT_RADD: if (s_axi_lite_arvalid) begin readAddr <= s_axi_lite_araddr[2+:REG_FILE_AWIDTH]; readState <= AWAIT_FETCH; end
            AWAIT_FETCH: readState <= AWAIT_READ;
            AWAIT_READ: if (s_axi_lite_rready) readState <= AWAIT_RADD;
            default: readState <= AWAIT_RADD;
        endcase
    end
    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid = (readState == AWAIT_READ);
    assign s_axi_lite_rdata = readData;

    // AXI-Lite Write Logic (Unchanged and Correct)
    always @(posedge s_axi_lite_aclk) begin
        if (!axi_resetn) writeState <= AWAIT_WADD_AND_DATA;
        else case (writeState)
            AWAIT_WADD_AND_DATA: case ({s_axi_lite_awvalid, s_axi_lite_wvalid})
                2'b10: begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WDATA; end
                2'b01: begin writeData <= s_axi_lite_wdata; writeState <= AWAIT_WADD; end
                2'b11: begin writeData <= s_axi_lite_wdata; writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE; end
                default: writeState <= AWAIT_WADD_AND_DATA;
            endcase
            AWAIT_WDATA: if (s_axi_lite_wvalid) begin writeData <= s_axi_lite_wdata; writeState <= AWAIT_WRITE; end
            AWAIT_WADD: if (s_axi_lite_awvalid) begin writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH]; writeState <= AWAIT_WRITE; end
            AWAIT_WRITE: begin regfile[writeAddr] <= writeData; writeState <= AWAIT_RESP; end
            AWAIT_RESP: if (s_axi_lite_bready) writeState <= AWAIT_WADD_AND_DATA;
            default: writeState <= AWAIT_WADD_AND_DATA;
        endcase
    end
    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;

    //================================================================
    // Core Processing and Display Logic
    //================================================================

    // 1. Control Signals
    logic start_convolution;
    logic write_done;
    logic write_enable_a;
    logic [11:0] bram_addr_a_ps_internal;

    logic unused_channel_debug_out;
    logic [15:0] max;
    logic [$clog2(10)-1:0] index;

    logic final_valid_out;

    // Synchronize the start signal from AXI clock domain to peripheral clock domain
    logic start_conv_sync1, start_conv_sync2;
    always_ff @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            start_conv_sync1 <= 1'b0;
            start_conv_sync2 <= 1'b0;
        end else begin
            start_conv_sync1 <= regfile[0][0]; // Read start bit from AXI register 0
            start_conv_sync2 <= start_conv_sync1;
        end
    end
    assign start_convolution = start_conv_sync2; // FIXED: Use the synchronized start signal

    // (* DONT_TOUCH = "true" *)
    top_capture #(
        .DATA_WIDTH(16),
        .KERNEL_SIZE_L1(5),
        .STRIDE(1),
        .PADDING(1),
        .CONV_OUTPUT(16),
        .IMAGE_SIZE(28),
        .OUTPUT_CHANNELS_L1(4),
        .OUTPUT_COL_SIZE_L1(24),
        .KERNEL_SIZE_L2(3),
        .OUTPUT_CHANNELS_L2(8),
        .INPUT_COL_SIZE_L2(12)
    ) top_capture_inst (
        .out_stream_aclk(out_stream_aclk),
        .periph_resetn(periph_resetn),
        .start(start_convolution),

        // Connections to Local BRAM (Port A - Write)
        .bram_addr_a(bram_addr_a),
        .bram_wrdata_a(bram_wrdata_a),
        .bram_we_a(write_enable_a),      // NOTE: This is an internal signal now
        .write_done(write_done),         // NOTE: This signal indicates processing is complete

        // Connections to PS BRAM (Read)
        .bram_rddata_a_ps(bram_rddata_a_ps),
        .bram_addr_a_ps(bram_addr_a_ps_internal),

        .unused_channel_debug_out(unused_channel_debug_out),

        .max(max),
        .index(index),

        .final_valid_out(final_valid_out)
    );

    assign bram_we_a = {32{write_enable_a}}; // Connect internal WE to 32-bit top-level port
    assign bram_addr_a_ps = bram_addr_a_ps_internal; // FIXED: Connect internal PS address to top-level port

    // 3. Display Pipeline: Local BRAM -> Tiler -> Packer -> Video Out
    wire valid_int = 1'b1;
    logic [7:0] current_gray_pixel;
    logic ready, first, lastx, lasty;
    reg [9:0] x;
    reg [8:0] y;
    
    // The Tiler reads the processed image from the Local BRAM and generates pixel coordinates
    top_tiler #(
        .X_SIZE(X_SIZE), .Y_SIZE(Y_SIZE), .READ_WIDTH(256), .PIX_BITS(8)
    ) top_tiler_inst (
        .out_stream_aclk(out_stream_aclk), 
        .periph_resetn(periph_resetn),
        .ready(ready), 
        .valid_int(valid_int), 
        .first(first), 
        .pixel(current_gray_pixel),
        .lastx(lastx), 
        .lasty(lasty), 
        .x(x), 
        .y(y),
        .bram_rdata(bram_rddata_b), // Read from Local BRAM Port B
        .bram_addr(bram_addr_b)     // Drive Local BRAM Port B Address
    );

    // 4. Final Output Formatting
    wire [7:0] r, g, b;
    assign r = current_gray_pixel;
    assign g = current_gray_pixel; 
    assign b = current_gray_pixel;

    packer pixel_packer(
        .aclk(out_stream_aclk), 
        .aresetn(periph_resetn),
        .r(r), .g(g), .b(b), 
        .eol(lastx), 
        .in_stream_ready(ready), 
        .valid(valid_int), .sof(first),
        .out_stream_tdata(out_stream_tdata), 
        .out_stream_tkeep(out_stream_tkeep),
        .out_stream_tlast(out_stream_tlast), 
        .out_stream_tready(out_stream_tready),
        .out_stream_tvalid(out_stream_tvalid), 
        .out_stream_tuser(out_stream_tuser)
    );

    //================================================================
    // BRAM Control Signal Tie-offs
    //================================================================
    
    // Local BRAM (Dual Port: A=Write, B=Read)
    assign bram_clk_a = out_stream_aclk;
    assign bram_en_a  = 1'b1;
    assign bram_rst_a = !periph_resetn;
    assign bram_clk_b = out_stream_aclk;
    assign bram_en_b  = 1'b1;
    assign bram_rst_b = !periph_resetn;
    assign bram_wrdata_b = '0; // Port B is read-only from PL side
    assign bram_we_b     = '0; // Port B is read-only from PL side

    // PS BRAM (Single Port: A=Read)
    assign bram_clk_a_ps = out_stream_aclk;
    assign bram_en_a_ps  = 1'b1;
    assign bram_rst_a_ps = !periph_resetn;
    assign bram_wrdata_a_ps = '0; // PS BRAM is read-only from PL side
    assign bram_we_a_ps     = '0; // PS BRAM is read-only from PL side
    
endmodule
