module top #(
    ADDR_W = 32
)(
    input logic             clk,
    input logic             rst,
    input logic             start_pulse
);



fsm #(
    .ADDR_W(ADDR_W)
) u_fsm (
    .clk(clk),
    .rst(rst),
    .cfg_start(start_pulse),
    .n_layer(32'h00000001), // TODO: set this from the CPU
    .dma_done(dma_done),
    .conv_done(conv_done),
    .irq_done(irq_done),
    .dma_go(dma_go),
    .conv_start(conv_start),
    .rd_addr(rd_addr),
    .wr_addr(wr_addr),
    .valid_in(valid_in),
    .sop(sop),
    .eop(eop)
);


axi_dma_0 u_axi_dma (
    .s_axi_lite_aclk     (clk),
    .m_axi_mm2s_aclk     (clk),
    .m_axi_s2mm_aclk     (clk),      // if used
    .axi_resetn          (!rst),     // active-low reset

    // AXI-Lite control
    .s_axi_lite_awaddr   (dma_awaddr),
    .s_axi_lite_awvalid  (dma_awvalid),
    .s_axi_lite_awready  (dma_awready),
    .s_axi_lite_wdata    (dma_wdata),
    .s_axi_lite_wvalid   (dma_wvalid),
    .s_axi_lite_wready   (dma_wready),
    .s_axi_lite_bresp    (dma_bresp),
    .s_axi_lite_bvalid   (dma_bvalid),
    .s_axi_lite_bready   (dma_bready),
    .s_axi_lite_araddr   (dma_araddr),
    .s_axi_lite_arvalid  (dma_arvalid),
    .s_axi_lite_arready  (dma_arready),
    .s_axi_lite_rdata    (dma_rdata),
    .s_axi_lite_rresp    (dma_rresp),
    .s_axi_lite_rvalid   (dma_rvalid),
    .s_axi_lite_rready   (dma_rready),

    // AXI-MM2S (read from memory)
    .m_axi_mm2s_araddr   (mm2s_araddr),
    .m_axi_mm2s_arlen    (mm2s_arlen),
    .m_axi_mm2s_arsize   (mm2s_arsize),
    .m_axi_mm2s_arburst  (mm2s_arburst),
    .m_axi_mm2s_arvalid  (mm2s_arvalid),
    .m_axi_mm2s_arready  (mm2s_arready),
    .m_axi_mm2s_rdata    (mm2s_rdata),
    .m_axi_mm2s_rresp    (mm2s_rresp),
    .m_axi_mm2s_rlast    (mm2s_rlast),
    .m_axi_mm2s_rvalid   (mm2s_rvalid),
    .m_axi_mm2s_rready   (mm2s_rready),

    // Stream output (MM2S to your MAC input buffer)
    .m_axis_mm2s_tdata   (dma_tdata),
    .m_axis_mm2s_tkeep   (dma_tkeep),
    .m_axis_mm2s_tvalid  (dma_tvalid),
    .m_axis_mm2s_tready  (dma_tready),
    .m_axis_mm2s_tlast   (dma_tlast),

    // Interrupt (optional)
    .mm2s_introut        (dma_done)
);


convolution #(
    .ADDR_W(ADDR_W)
) u_convolution (
    .clk(clk),
    .rst(rst),
    .kernel_load(kernel_load),
    .valid_in(valid_in),
    .valid_out(valid_out),
    .data_out(data_out)
);

endmodule