// ====================================================================
// Testbench for display_tiler module
// Simulates VTC timing and BRAM, outputs to file for visualization
// ====================================================================

`timescale 1ns/1ps

module tb_display_tiler;

    // Parameters matching the DUT
    localparam int TILE_W = 96;
    localparam int TILE_H = 96;
    localparam int COLS = 4;
    localparam int NM = 22;
    
    // Display parameters
    localparam int H_TOTAL = COLS * TILE_W;  // 384 pixels
    localparam int V_TOTAL = 6 * TILE_H;     // 576 pixels
    
    // Clock and reset
    logic pix_clk = 0;
    logic rst_n = 1;
    
    // BRAM interface
    logic [15:0] bram_addr;
    logic [7:0] bram_q;
    
    // VTC pixel counters
    logic [11:0] h_cnt = 0;
    logic [11:0] v_cnt = 0;
    
    // AXI-Stream outputs
    logic [23:0] tdata;
    logic tvalid;
    logic tlast;
    logic [3:0] tkeep;
    logic tuser;
    
    // Test BRAM - simulates memory with test patterns
    logic [7:0] test_bram [0:32767];  // 32k bytes
    
    // Clock generation
    always #5 pix_clk = ~pix_clk;  // 100MHz pixel clock
    
    // BRAM simulation
    assign bram_q = test_bram[bram_addr];
    
    // VTC counter simulation (generates h_cnt, v_cnt like VTC)
    always_ff @(posedge pix_clk) begin
        if (!rst_n) begin
            h_cnt <= 0;
            v_cnt <= 0;
        end else begin
            if (h_cnt == H_TOTAL - 1) begin
                h_cnt <= 0;
                if (v_cnt == V_TOTAL - 1) begin
                    v_cnt <= 0;
                end else begin
                    v_cnt <= v_cnt + 1;
                end
            end else begin
                h_cnt <= h_cnt + 1;
            end
        end
    end
    
    // DUT instantiation
    display_tiler #(
        .TILE_W(TILE_W),
        .TILE_H(TILE_H),
        .COLS(COLS),
        .NM(NM)
    ) dut (
        .pix_clk(pix_clk),
        .bram_addr(bram_addr),
        .bram_q(bram_q),
        .h_cnt(h_cnt),
        .v_cnt(v_cnt),
        .tdata(tdata),
        .tvalid(tvalid),
        .tlast(tlast),
        .tkeep(tkeep),
        .tuser(tuser)
    );
    
    // File output for visualization
    integer outfile;
    logic [7:0] r, g, b;
    assign {r, g, b} = tdata;
    
    initial begin
        // Initialize test BRAM with patterns
        init_test_bram();
        
        // Open output file
        outfile = $fopen("display_output.ppm", "w");
        $fwrite(outfile, "P3\n%0d %0d\n255\n", H_TOTAL, V_TOTAL);
        
        // Reset
        rst_n = 0;
        #100;
        rst_n = 1;
        
        $display("Starting display tiler test...");
        $display("Resolution: %0dx%0d", H_TOTAL, V_TOTAL);
        $display("Tile size: %0dx%0d", TILE_W, TILE_H);
        
        // Wait for one complete frame
        wait(tuser == 1);  // Start of frame
        $display("Frame started at time %0t", $time);
        
        // Capture one frame
        repeat(H_TOTAL * V_TOTAL) begin
            @(posedge pix_clk);
            if (tvalid) begin
                $fwrite(outfile, "%0d %0d %0d ", r, g, b);
                if (tlast) $fwrite(outfile, "\n");
            end
        end
        
        $fclose(outfile);
        $display("Frame capture complete. Output saved to display_output.ppm");
        $finish;
    end
    
    // Initialize test BRAM with recognizable patterns
    task init_test_bram();
        automatic int addr;
        automatic logic [7:0] colors[4] = '{8'h40, 8'h80, 8'hC0, 8'hFF};
        
        // Clear memory
        for (int i = 0; i < 32768; i++) begin
            test_bram[i] = 8'h00;
        end
        
        // Tile 0: Input image (28x28) - checkerboard pattern
        for (int y = 0; y < 28; y++) begin
            for (int x = 0; x < 28; x++) begin
                addr = 16'h0000 + y*28 + x;
                test_bram[addr] = ((x+y) % 2) ? 8'hFF : 8'h80;
            end
        end
        
        // Tiles 1-4: Conv1 maps (24x24) - gradient patterns
        addr = 16'h0320; // Tile 1
        for (int i = 0; i < 576; i++) test_bram[addr + i] = (i % 24) * 10;
        
        addr = 16'h0560; // Tile 2  
        for (int i = 0; i < 576; i++) test_bram[addr + i] = (i / 24) * 10;
        
        addr = 16'h07A0; // Tile 3
        for (int i = 0; i < 576; i++) test_bram[addr + i] = 8'h80;
        
        addr = 16'h09E0; // Tile 4
        for (int i = 0; i < 576; i++) test_bram[addr + i] = 8'hC0;
        
        // Tiles 17-20: Conv2 maps (8x8) - solid colors
        for (int tile = 0; tile < 4; tile++) begin
            addr = 16'h2720 + tile * 16'h40;
            for (int i = 0; i < 64; i++) begin
                test_bram[addr + i] = colors[tile];
            end
        end
        
        // Tile 21: Probability bar (10x1) - ascending values
        for (int i = 0; i < 10; i++) begin
            test_bram[16'h2820 + i] = (i * 255) / 9;
        end
        
        $display("Test BRAM initialized with patterns");
    endtask
    
    // Monitor for debugging
    always @(posedge pix_clk) begin
        if (tuser) $display("SOF at (%0d,%0d)", h_cnt, v_cnt);
        if (tlast) $display("EOL at (%0d,%0d)", h_cnt, v_cnt);
    end

endmodule
