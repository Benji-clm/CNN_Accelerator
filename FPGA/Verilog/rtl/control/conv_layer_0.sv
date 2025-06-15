module conv_layer_0 #(
    parameter DATA_WIDTH = 16,
    parameter IMAGE_SIZE = 28,
    parameter KERNEL_SIZE = 5,
    parameter NUM_KERNELS = 4,
    parameter STRIDE = 1,
    parameter PADDING = 1
)(
    input logic clk,
    input logic rst,
    input logic start,
    output logic [DATA_WIDTH-1:0] data_out [0:(IMAGE_SIZE-KERNEL_SIZE)* NUM_KERNELS-1],
    output logic done
);

logic done_0, done_1, done_2, done_3;

control_cv5 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .IMAGE_SIZE(IMAGE_SIZE),
    .KERNEL_NUM(0), // This is the first kernel
    .STRIDE(STRIDE),
    .PADDING(PADDING)
) cv5_inst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_out(data_out[0 +: (IMAGE_SIZE-KERNEL_SIZE)]), // Output for the first kernel
    .done(done_0)
);

// Instantiate additional control modules for other kernels
control_cv5 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .IMAGE_SIZE(IMAGE_SIZE),
    .KERNEL_NUM(1), // Second kernel
    .STRIDE(STRIDE),
    .PADDING(PADDING)
) cv5_inst_1 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_out(data_out[(IMAGE_SIZE-KERNEL_SIZE) +: (IMAGE_SIZE-KERNEL_SIZE)]), // Output for the second kernel
    .done(done_1)
);

control_cv5 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .IMAGE_SIZE(IMAGE_SIZE),
    .KERNEL_NUM(2), // Third kernel
    .STRIDE(STRIDE),
    .PADDING(PADDING)
) cv5_inst_2 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_out(data_out[2*(IMAGE_SIZE-KERNEL_SIZE) +: (IMAGE_SIZE-KERNEL_SIZE)]), // Output for the third kernel
    .done(done_2)
);

control_cv5 #(
    .DATA_WIDTH(DATA_WIDTH),
    .KERNEL_SIZE(KERNEL_SIZE),
    .IMAGE_SIZE(IMAGE_SIZE),
    .KERNEL_NUM(3), // Fourth kernel
    .STRIDE(STRIDE),
    .PADDING(PADDING)
) cv5_inst_3 (
    .clk(clk),
    .rst(rst),
    .start(start),
    .data_out(data_out[3*(IMAGE_SIZE-KERNEL_SIZE) +: (IMAGE_SIZE-KERNEL_SIZE)]), // Output for the fourth kernel
    .done(done_3)
);

assign done = done_0 && done_1 && done_2 && done_3;
endmodule
