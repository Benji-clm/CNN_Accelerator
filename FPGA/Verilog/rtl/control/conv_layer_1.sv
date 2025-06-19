module conv_layer_1 #(
    parameter DATA_WIDTH              = 16,
    parameter KERNEL_SIZE             = 3,
    parameter INPUT_COL_SIZE          = 12,
    parameter NUM_CHANNELS            = 8,
    parameter INPUT_CHANNEL_NUMBER    = 4
)(
    input logic clk,
    input logic rst,

    // --- Control Signals ---
    input logic valid_in,

    // --- Data Inputs ---
    input logic signed [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],

    // Feature map
    output logic signed [DATA_WIDTH-1:0] fm_columns [NUM_CHANNELS-1:0][INPUT_COL_SIZE - KERNEL_SIZE :0],
    // --- Data Outputs ---
    output logic signed [DATA_WIDTH-1:0] output_columns[NUM_CHANNELS-1:0][(INPUT_COL_SIZE - KERNEL_SIZE + 1) / 2 - 1:0],
    output logic valid_out,
    output logic column_valid_out
);

    // --- Hardcoded Kernels and Biases ---
    localparam logic signed [DATA_WIDTH-1:0] KERNELS [NUM_CHANNELS-1:0] [INPUT_CHANNEL_NUMBER-1:0] [KERNEL_SIZE * KERNEL_SIZE-1:0] =
    '{
        // --- Filter 0 Kernels ---
        '{ // Output Channel 0
            '{16'h0BAE, 16'h0622, 16'hF86F, 16'hE4C6, 16'h0FF9, 16'hFEB4, 16'h1863, 16'hEE25, 16'hFCF2}, // Input Chan 0
            '{16'h0382, 16'h09D0, 16'hF421, 16'hE964, 16'hE84D, 16'h116F, 16'hE33F, 16'h0AE3, 16'h1417}, // Input Chan 1
            '{16'h0F1B, 16'h15F6, 16'hFF7E, 16'h021E, 16'hFB32, 16'hF419, 16'hB83A, 16'hD810, 16'hE93B}, // Input Chan 2
            '{16'hEC81, 16'h0A04, 16'h11B7, 16'h0992, 16'h1BF8, 16'hFCA9, 16'hF2FF, 16'hEA74, 16'h0DB1}  // Input Chan 3
        },

        // --- Filter 1 Kernels ---
        '{ // Output Channel 1
            '{16'hFA5A, 16'h05D4, 16'hF765, 16'h197C, 16'h0C16, 16'h0E8E, 16'h0762, 16'h05AB, 16'hF999}, // Input Chan 0
            '{16'hD99F, 16'h18F9, 16'hF527, 16'hD915, 16'hFCF9, 16'h1700, 16'hFD10, 16'hFD60, 16'h0D4A}, // Input Chan 1
            '{16'h0111, 16'hF51C, 16'h11C6, 16'hEF3D, 16'hEE76, 16'h13F6, 16'hE9CA, 16'hE5A7, 16'hFD87}, // Input Chan 2
            '{16'h0E60, 16'hF2FA, 16'hF69A, 16'h03D7, 16'hF86E, 16'hFF96, 16'h03A2, 16'hFC45, 16'h0FDA}  // Input Chan 3
        },

        // --- Filter 2 Kernels ---
        '{ // Output Channel 2
            '{16'h030D, 16'hF3E7, 16'h0C0E, 16'h0AF9, 16'h03F2, 16'hD73C, 16'h1A13, 16'h0F6C, 16'hE20A}, // Input Chan 0
            '{16'hF550, 16'hE396, 16'hF51A, 16'hF77D, 16'hE90F, 16'hFBF2, 16'hEC5F, 16'hF05F, 16'h0292}, // Input Chan 1
            '{16'hFD12, 16'hF485, 16'h0F92, 16'h11DD, 16'h0153, 16'hFF12, 16'h096A, 16'h0EB1, 16'h0991}, // Input Chan 2
            '{16'h0C4F, 16'hDF03, 16'hF0EB, 16'h0AF9, 16'hF110, 16'hC6D4, 16'h21FA, 16'hF584, 16'hC9FB}  // Input Chan 3
        },

        // --- Filter 3 Kernels ---
        '{ // Output Channel 3
            '{16'h0185, 16'hFAA0, 16'h088A, 16'h07F0, 16'h0753, 16'h0A3F, 16'hEB8C, 16'hF251, 16'h13A3}, // Input Chan 0
            '{16'hF90C, 16'h0B76, 16'h013C, 16'h0BA1, 16'h0A22, 16'hFF77, 16'hFFE7, 16'h1399, 16'hF23A}, // Input Chan 1
            '{16'hE0E9, 16'hE1B5, 16'hC1DC, 16'hF817, 16'hFA0F, 16'hDBB8, 16'h184D, 16'h2F5E, 16'h13A1}, // Input Chan 2
            '{16'h1143, 16'hEF0B, 16'hECC2, 16'h12A9, 16'h01D4, 16'hE6FB, 16'hFCCE, 16'hD848, 16'h1AA2}  // Input Chan 3
        },

        // --- Filter 4 Kernels ---
        '{ // Output Channel 4
            '{16'h0711, 16'h09A0, 16'hFA55, 16'hDCDA, 16'hDEC5, 16'hEB01, 16'hF437, 16'hEA2D, 16'hED32}, // Input Chan 0
            '{16'hE9A8, 16'hF26D, 16'hE8A1, 16'h0E71, 16'h0A81, 16'hEA2D, 16'h0DBC, 16'h0B77, 16'h09F5}, // Input Chan 1
            '{16'hFAAA, 16'hF98A, 16'h0CD6, 16'hE721, 16'hEFE9, 16'hF0FE, 16'h104D, 16'h09E7, 16'hFDE1}, // Input Chan 2
            '{16'hE92E, 16'h01CC, 16'h1CE4, 16'hF87B, 16'h1076, 16'h1E26, 16'hDAF3, 16'hF1D3, 16'h03A5}  // Input Chan 3
        },

        // --- Filter 5 Kernels ---
        '{ // Output Channel 5
            '{16'hDCBB, 16'hE6BA, 16'h0B43, 16'hE77B, 16'hD49E, 16'h0910, 16'hECAB, 16'h0262, 16'h1C92}, // Input Chan 0
            '{16'h0CCC, 16'hFD65, 16'hF13C, 16'h10D2, 16'hF684, 16'hE511, 16'h07B8, 16'hE4D7, 16'hDA6F}, // Input Chan 1
            '{16'hF876, 16'h1694, 16'hEC4A, 16'hFC10, 16'h09A4, 16'hCFDB, 16'h1FDC, 16'h09FC, 16'hF3D8}, // Input Chan 2
            '{16'hF022, 16'h0855, 16'hFCD8, 16'h0B2A, 16'hFE20, 16'hF7BE, 16'h0AE0, 16'hF43E, 16'hFA99}  // Input Chan 3
        },

        // --- Filter 6 Kernels ---
        '{ // Output Channel 6
            '{16'h08FA, 16'hFE4E, 16'hF7C9, 16'hE774, 16'h0B9E, 16'h133D, 16'h0661, 16'hF353, 16'h05AE}, // Input Chan 0
            '{16'h0756, 16'h0DB9, 16'h0570, 16'h0E76, 16'h0E4E, 16'h04C1, 16'h20E3, 16'hF4A9, 16'h0368}, // Input Chan 1
            '{16'hEA05, 16'hF940, 16'h07FF, 16'h1431, 16'hF767, 16'hF3CC, 16'h0D5A, 16'h0341, 16'hE752}, // Input Chan 2
            '{16'hF466, 16'hF75D, 16'hFCB5, 16'h0AE0, 16'hEEFE, 16'hDC4F, 16'hE2C4, 16'h0E8C, 16'hE3F9}  // Input Chan 3
        },

        // --- Filter 7 Kernels ---
        '{ // Output Channel 7
            '{16'hE0AD, 16'hFB81, 16'h0231, 16'hDC06, 16'h0F71, 16'h0862, 16'h0AD6, 16'h12E4, 16'h00C2}, // Input Chan 0
            '{16'h0ED5, 16'hFC2A, 16'hE841, 16'h12AF, 16'hFBEB, 16'hD460, 16'hF1A7, 16'hE705, 16'hEF61}, // Input Chan 1
            '{16'hFDE9, 16'hF461, 16'hF45E, 16'hFE68, 16'hFC53, 16'hF2D4, 16'hFBCC, 16'hF123, 16'hD734}, // Input Chan 2
            '{16'h0483, 16'h0082, 16'h2668, 16'hEB23, 16'hF39B, 16'h06DE, 16'hF7AB, 16'h035F, 16'h0B9B}  // Input Chan 3
        }
    };
    localparam logic signed [DATA_WIDTH-1:0] BIASES[0:NUM_CHANNELS-1] = '{16'hF72D, 16'hF80D, 16'hEF95, 16'hE994, 16'h0254, 16'h0864, 16'h0684, 16'hEF1F};

    // Internal Signals
    logic kernel_load_r;
    logic channel_valid_in;
    typedef enum logic [1:0] {IDLE, LOAD, RUN} state_t;
    state_t state, next_state;
    logic [1:0] load_cycle_count;
    logic signed [DATA_WIDTH-1:0] kernel_wires [0:NUM_CHANNELS-1][0:INPUT_CHANNEL_NUMBER-1][0:KERNEL_SIZE-1];
    logic [NUM_CHANNELS-1:0] valid_out_wires;
    logic [NUM_CHANNELS-1:0] column_valid_wires;

    // Kernel Loading State Machine
    always_ff @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            load_cycle_count <= '0;
        end else begin
            state <= next_state;
            if (state == LOAD) begin
                load_cycle_count <= load_cycle_count + 1;
            end
        end
    end

    always_comb begin
        next_state = state;
        kernel_load_r = 1'b0;
        case(state)
            IDLE: next_state = LOAD;
            LOAD: begin
                kernel_load_r = 1'b1;
                if (load_cycle_count == KERNEL_SIZE - 1) begin
                    next_state = RUN;
                end
            end
            RUN: next_state = RUN;
        endcase
    end

    assign channel_valid_in = valid_in | kernel_load_r;

    // Kernel Muxing Logic
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            for (int f = 0; f < INPUT_CHANNEL_NUMBER; f++) begin
                for (int c = 0; c < KERNEL_SIZE; c++) begin
                    kernel_wires[ch][f][c] = KERNELS[ch][f][c*KERNEL_SIZE + load_cycle_count];
                end
            end
        end
    end

    // Channel Instantiation
    generate
        for (genvar ch_idx = 0; ch_idx < NUM_CHANNELS; ch_idx++) begin : gen_channel
            logic signed [DATA_WIDTH - 1:0] channel_output [INPUT_COL_SIZE - KERNEL_SIZE:0];
            logic signed [DATA_WIDTH - 1:0] ReLU_output [INPUT_COL_SIZE - KERNEL_SIZE:0];

            cv3_channel #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE),
                .INPUT_COL_SIZE(INPUT_COL_SIZE),
                .INPUT_CHANNEL_NUMBER(INPUT_CHANNEL_NUMBER),
                .BIAS(BIASES[ch_idx]) // **FIXED**: Specialize via parameter override
            ) u_cv3_channel (
                .clk(clk), .rst(rst), .kernel_load(kernel_load_r), .valid_in(channel_valid_in),
                .input_columns(input_columns), .kernel_inputs(kernel_wires[ch_idx]),
                .output_column(channel_output), .valid_out(column_valid_wires[ch_idx])
            );

            assign fm_columns[ch_idx] = channel_output;

            ReLU_column #(.COLUMN_SIZE(INPUT_COL_SIZE - KERNEL_SIZE + 1))
            ReLU (.data_in(channel_output), .data_out(ReLU_output));

            pooling_layer #(.WINDOWS((INPUT_COL_SIZE - KERNEL_SIZE + 1)/2))
            Pooling (
                .clk(clk), .rst(rst), .valid_in(column_valid_wires[ch_idx]),
                .input_column(ReLU_output), .valid_out(valid_out_wires[ch_idx]),
                .output_column(output_columns[ch_idx])
            );
        end
    endgenerate

    always_comb begin
        logic v_out_reduced = 1'b1;
        logic c_valid_reduced = 1'b1;
        for (int i = 0; i < NUM_CHANNELS; i++) begin
            v_out_reduced = v_out_reduced & valid_out_wires[i];
            c_valid_reduced = c_valid_reduced & column_valid_wires[i];
        end
        valid_out = v_out_reduced;
        column_valid_out = c_valid_reduced;
    end

endmodule