    module pixel_generator(
    input           out_stream_aclk,
    input           s_axi_lite_aclk,
    input           axi_resetn,
    input           periph_resetn,

    //Stream output
    output [31:0]   out_stream_tdata,
    output [3:0]    out_stream_tkeep,
    output          out_stream_tlast,
    input           out_stream_tready,
    output          out_stream_tvalid,
    output [0:0]    out_stream_tuser, 

    //AXI-Lite S
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

    // BRAM SIGNALLLLSSSSS
    input [255:0]			bram_rddata_a,
    output [11:0]			bram_addr_a, // address
    output 					bram_clk_a, // clock
    output [255:0]			bram_wrdata_a, // written DATA into BRAM
    output					bram_en_a, // global enable
    output 					bram_rst_a, // reset	
    output [3:0]			bram_we_a, // write enable for each byte

    // BRAM SIGNALLLLSSSSS - port B
    input [255:0]			bram_rddata_b,
    output [11:0]			bram_addr_b, // address
    output 					bram_clk_b, // clock
    output [255:0]			bram_wrdata_b, // written DATA into BRAM
    output					bram_en_b, // global enable
    output 					bram_rst_b, // reset	
    output [3:0]			bram_we_b, // write enable for each byte


    // BRAM Processing System
    input [31:0]			bram_rddata_a_ps,
    output [11:0]			bram_addr_a_ps, // address
    output 					bram_clk_a_ps, // clock
    output [31:0]			bram_wrdata_a_ps, // written DATA into BRAM
    output					bram_en_a_ps, // global enable
    output 					bram_rst_a_ps, // reset	
    output [3:0]			bram_we_a_ps // write enable for each byte
    );

    localparam X_SIZE = 640;
    localparam Y_SIZE = 480;
    parameter  REG_FILE_SIZE = 8;
    localparam REG_FILE_AWIDTH = $clog2(REG_FILE_SIZE);
    parameter  AXI_LITE_ADDR_WIDTH = 8;

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

    reg [31:0]                          regfile [REG_FILE_SIZE-1:0];
    reg [REG_FILE_AWIDTH-1:0]           writeAddr, readAddr;
    reg [31:0]                          readData, writeData;
    reg [1:0]                           readState = AWAIT_RADD;
    reg [2:0]                           writeState = AWAIT_WADD_AND_DATA;

    //Read from the register file
    always @(posedge s_axi_lite_aclk) begin
        
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

    assign s_axi_lite_arready = (readState == AWAIT_RADD);
    assign s_axi_lite_rresp = (readAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;
    assign s_axi_lite_rvalid = (readState == AWAIT_READ);
    assign s_axi_lite_rdata = readData;

    //Write to the register file, use a state machine to track address write, data write and response read events
    always @(posedge s_axi_lite_aclk) begin

        if (!axi_resetn) begin
            writeState <= AWAIT_WADD_AND_DATA;
        end

        else case (writeState)

            AWAIT_WADD_AND_DATA: begin  //Idle, awaiting a write address or data
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

            AWAIT_WDATA: begin //Received address, waiting for data
                if (s_axi_lite_wvalid) begin
                    writeData <= s_axi_lite_wdata;
                    writeState <= AWAIT_WRITE;
                end
            end

            AWAIT_WADD: begin //Received data, waiting for address
                if (s_axi_lite_awvalid) begin
                    writeAddr <= s_axi_lite_awaddr[2+:REG_FILE_AWIDTH];
                    writeState <= AWAIT_WRITE;
                end
            end

            AWAIT_WRITE: begin //Perform the write
                regfile[writeAddr] <= writeData;
                writeState <= AWAIT_RESP;
            end

            AWAIT_RESP: begin //Wait to send response
                if (s_axi_lite_bready) begin
                    writeState <= AWAIT_WADD_AND_DATA;
                end
            end

            default: begin
                writeState <= AWAIT_WADD_AND_DATA;
            end
        endcase
    end

    assign s_axi_lite_awready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WADD);
    assign s_axi_lite_wready = (writeState == AWAIT_WADD_AND_DATA || writeState == AWAIT_WDATA);
    assign s_axi_lite_bvalid = (writeState == AWAIT_RESP);
    assign s_axi_lite_bresp = (writeAddr < REG_FILE_SIZE) ? AXI_OK : AXI_ERR;



    reg [9:0] x;
    reg [8:0] y;

    wire first = (x == 0) & (y == 0);
    wire lastx = (x == 23);  // Last pixel in 24-pixel row
    wire lasty = (y == 23);  // Last row in 24-row column
    wire [7:0] frame = regfile[0];
    wire ready;

    // always @(posedge out_stream_aclk) begin
    //     if (periph_resetn) begin
    //         if (ready & valid_int) begin
    //             if (lastx) begin
    //                 x <= 9'd0;
    //                 if (lasty) y <= 9'd0;
    //                 else y <= y + 9'd1;
    //             end
    //             else x <= x + 9'd1;
    //         end
    //     end
    //     else begin
    //         x <= 0;
    //         y <= 0;
    //     end
    // end


    wire [7:0] r, g, b;

    // ________________________ IMAGE TEST ____________________

    wire [15:0] test_column_data [23:0];  // 2D array: 24 rows of 16-bit values
    wire test_valid_col;
    wire [7:0] current_gray_pixel;
    wire pixel_valid;

    wire [9:0] x_coordinate;
    reg [8:0] column_n; 

    // Generate test grayscale gradient data (white to black) in proper IEEE 754 FP16 format
    // FP16 format: sign(1) + exponent(5) + mantissa(10)
    genvar i;
    generate
        for (i = 0; i < 24; i = i + 1) begin : gen_gradient
            wire [15:0] gradient_value;
            
            case (i)
                0:  assign gradient_value = 16'h3C00;  // 1.0
                1:  assign gradient_value = 16'h3BDC;  // ~0.956
                2:  assign gradient_value = 16'h3BB8;  // ~0.913
                3:  assign gradient_value = 16'h3B94;  // ~0.870
                4:  assign gradient_value = 16'h3B70;  // ~0.826
                5:  assign gradient_value = 16'h3B4C;  // ~0.783
                6:  assign gradient_value = 16'h3B28;  // ~0.739
                7:  assign gradient_value = 16'h3B04;  // ~0.696
                8:  assign gradient_value = 16'h3AE0;  // ~0.652
                9:  assign gradient_value = 16'h3ABC;  // ~0.609
                10: assign gradient_value = 16'h3A98;  // ~0.565
                11: assign gradient_value = 16'h3A74;  // ~0.522
                12: assign gradient_value = 16'h3A50;  // ~0.478
                13: assign gradient_value = 16'h3A2C;  // ~0.435
                14: assign gradient_value = 16'h3A08;  // ~0.391
                15: assign gradient_value = 16'h39E4;  // ~0.348
                16: assign gradient_value = 16'h39C0;  // ~0.304
                17: assign gradient_value = 16'h399C;  // ~0.261
                18: assign gradient_value = 16'h3978;  // ~0.217
                19: assign gradient_value = 16'h3954;  // ~0.174
                20: assign gradient_value = 16'h3930;  // ~0.130
                21: assign gradient_value = 16'h390C;  // ~0.087
                22: assign gradient_value = 16'h38E8;  // ~0.043
                23: assign gradient_value = 16'h0000;  // 0.0
                default: assign gradient_value = 16'h0000;
            endcase
            
            assign test_column_data[i] = gradient_value;
        end
    endgenerate

    // Generate test valid signal - pulse every 28 cycles
    reg [4:0] test_counter;
    always @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            test_counter <= 0;
        end else begin
            test_counter <= (test_counter == 27) ? 0 : test_counter + 1;
        end
    end
    assign test_valid_col = (test_counter == 0);

    easy_image #(
        .PIX_W(24),
        .PIX_H(24),
        .X_SIZE(X_SIZE)
    ) image_output (
        .clk(out_stream_aclk),
        .rst(!periph_resetn),  // Note: rst is active high in easy_image
        .valid_col(test_valid_col),
        .data_col(test_column_data),  // Corrected port name
        .current_gray_pixel(current_gray_pixel),
        .x_coordinate(x_coordinate),  // Corrected port name
        .pixel_valid(pixel_valid)
    );

    // Use the grayscale pixel output
    assign r = current_gray_pixel;
    assign g = current_gray_pixel;
    assign b = current_gray_pixel;



    always @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            column_n <= 0;
        end else begin
            // When easy_image finishes outputting a column (pixel_valid goes low after being high)
            if (test_valid_col) begin
                // New column data is being loaded, increment column counter
                column_n <= (column_n == 23) ? 0 : column_n + 1;
            end
        end
    end

    // Use column_n as y-coordinate and x_coordinate from easy_image as x-coordinate
    always @(posedge out_stream_aclk) begin
        if (!periph_resetn) begin
            x <= 0;
            y <= 0;
        end else begin
            if (pixel_valid) begin
                x <= x_coordinate; 
                y <= column_n;
            end
        end
    end


    // _________________________________________________________


    wire valid_int = 1'b1;


    // parameter [255:0] TEST_DATA      = 256'hCAFEBEEF_D15EA5ED_DEADBEEF_01234567_89ABCDEF_FEEDFACE_F00DBABE_0BADF00D;
    // parameter [11:0]  TEST_INT_ADDR  = 12'h000;   // location in 256-bit BRAM
    // parameter [11:0]  TEST_PS_BASE   = 12'h100;   // first 32-bit word address in PS BRAM

    // assign bram_wrdata_a = TEST_DATA;
    // assign bram_en_a = 1;
    // assign bram_addr_a = TEST_INT_ADDR;
    // assign bram_clk_a = out_stream_aclk;
    // assign bram_rst_a = 0;

    // localparam IDLE   = 1'b0;
    // localparam ACTIVE = 1'b1;

    // reg state;
    // reg [3:0] write_enable_reg;

    // always @(posedge out_stream_aclk or negedge axi_resetn) begin
    // 	if (!axi_resetn) begin
    // 		state <= IDLE;
    // 		write_enable_reg <= 4'b1111;
    //     end else begin
    // 		case (state)
    // 			IDLE: begin
    //                 if(regfile[2] == 1'b1) begin
    //                     state <= ACTIVE; 
    //                     write_enable_reg <= 4'b1111;
    //                 end else state <= ACTIVE;
    // 			end
    // 			ACTIVE: begin
    // 				state <= IDLE;
    // 				write_enable_reg <= 4'b0000;
    // 			end
    // 		endcase
    // 	end
    // end

    // assign bram_we_a = write_enable_reg;

    // reg [2:0] state_PS_BRAM;
    // reg [11:0] PS_address;

    // assign bram_addr_a_ps = PS_address; // to increment at each state
    // assign bram_clk_a_ps = out_stream_aclk;
    // assign bram_en_a_ps = 1;
    // assign bram_rst_a_ps = 0;

    // // State encoding - each representing 32 bits within the PS BRAM
    // localparam S0 = 4'd0;
    // localparam S1 = 4'd1;
    // localparam S2 = 4'd2;
    // localparam S3 = 4'd3;
    // localparam S4 = 4'd4;
    // localparam S5 = 4'd5;
    // localparam S6 = 4'd6;
    // localparam S7 = 4'd7;
    // localparam DONE = 4'd8;

    // reg [31:0] write_PS_value;
    // reg [3:0]  write_en_PS;

    // always @(posedge out_stream_aclk or negedge axi_resetn) begin
    //     if (!axi_resetn) begin
    //         state_PS_BRAM <= S0;
    //         PS_address <= TEST_PS_BASE;
    //         write_en_PS <= 4'b0000;
    //     end else begin
    //         case (state_PS_BRAM)
    //             S0: begin
    //                 if(regfile[2] == 1'b1) begin
    //                     write_en_PS <= 4'b1111;
    //                     state_PS_BRAM <= S1;
    //                     write_PS_value <= bram_rddata_a[31:0];
    //                     PS_address <= TEST_PS_BASE;
    //                 end else begin
    //                     write_en_PS <= 4'b0000;
    //                     state_PS_BRAM <= S0;
    //                 end
    //             end
    //             S1: begin
    //                 state_PS_BRAM <= S2;
    //                 write_PS_value <= bram_rddata_a[63:32];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;  // Enable write for next cycle
    //             end
    //             S2: begin
    //                 state_PS_BRAM <= S3;
    //                 write_PS_value <= bram_rddata_a[95:64];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             S3: begin
    //                 state_PS_BRAM <= S4;
    //                 write_PS_value <= bram_rddata_a[127:96];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             S4: begin
    //                 state_PS_BRAM <= S5;
    //                 write_PS_value <= bram_rddata_a[159:128];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             S5: begin
    //                 state_PS_BRAM <= S6;
    //                 write_PS_value <= bram_rddata_a[191:160];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             S6: begin
    //                 state_PS_BRAM <= S7;
    //                 write_PS_value <= bram_rddata_a[223:192];
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             S7: begin
    //                 write_PS_value <= bram_rddata_a[255:224];
    //                 state_PS_BRAM <= DONE;
    //                 PS_address <= PS_address + 12'd4;
    //                 write_en_PS <= 4'b1111;
    //             end
    //             DONE: begin 
    //                 state_PS_BRAM <= S0;
    //                 write_en_PS <= 4'b0000;
    //             end
    //             default: begin
    //                 write_en_PS <= 4'b0000;
    //                 state_PS_BRAM <= S0;
    //                 PS_address <= TEST_PS_BASE;
    //             end
    //         endcase
    //     end
    // end

    // assign bram_we_a_ps = write_en_PS;
    // assign bram_wrdata_a_ps = write_PS_value;


    // conv_5 convolution_n0_5(
    //     .clk        (out_stream_aclk),
    //     .rst        (!periph_resetn),
    //     .data_in0   (bram_rddata_a[15:0]),
    //     .data_in1   (bram_rddata_a[15:0]),
    //     .data_in2   (bram_rddata_a[15:0]),
    //     .data_in3   (bram_rddata_a[15:0]),
    //     .data_in4   (bram_rddata_a[15:0]),
    //     .data_out   (data_out),
    //     .kernel_load(kernel_load),
    //     .valid_in   (valid_in),
    //     .valid_out  (valid_out)
    // );



    packer pixel_packer(    .aclk(out_stream_aclk),
                            .aresetn(periph_resetn),
                            .r(r), .g(g), .b(b),
                            .eol(lastx), .in_stream_ready(ready), .valid(valid_int), .sof(first),
                            .out_stream_tdata(out_stream_tdata), .out_stream_tkeep(out_stream_tkeep),
                            .out_stream_tlast(out_stream_tlast), .out_stream_tready(out_stream_tready),
                            .out_stream_tvalid(out_stream_tvalid), .out_stream_tuser(out_stream_tuser) );

    

    // B port unused for now
    assign bram_addr_b = 12'h0;
    assign bram_clk_b = out_stream_aclk;
    assign bram_wrdata_b = 256'h0;
    assign bram_en_b = 1'b0;
    assign bram_rst_b = 1'b0;
    assign bram_we_b = 4'h0;

    assign bram_addr_a = 12'h0;
    assign bram_clk_a = out_stream_aclk;
    assign bram_wrdata_a = 256'h0;
    assign bram_en_a = 1'b0;
    assign bram_rst_a = 1'b0;
    assign bram_we_a = 4'h0;

    assign bram_addr_a_ps = 12'h0;
    assign bram_clk_a_ps = out_stream_aclk;
    assign bram_wrdata_a_ps = 256'h0;
    assign bram_en_a_ps = 1'b0;
    assign bram_rst_a_ps = 1'b0;
    assign bram_we_a_ps = 4'h0;


    endmodule
