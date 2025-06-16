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
    input logic [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],


    // --- Data Outputs ---
    output logic [DATA_WIDTH-1:0] output_columns[NUM_CHANNELS-1:0][(INPUT_COL_SIZE - KERNEL_SIZE + 1) / 2 - 1:0],
    output logic valid_out
);

    localparam [DATA_WIDTH-1:0] KERNELS[0:NUM_CHANNELS-1][0:INPUT_CHANNEL_NUMBER-1][0:KERNEL_SIZE*KERNEL_SIZE-1] =
    '{
        // --- Channel 0 Kernels (was Filter 1) ---
        '{
            '{16'h31d7, 16'h2e22, 16'haf91, 16'hb6ce, 16'h33fc, 16'ha532, 16'h3619, 16'hb477, 16'haa1c}, // Filter 0 (was Input Channel 1)
            '{16'h2b04, 16'h30e8, 16'hb1ef, 16'hb5a7, 16'hb5ed, 16'h345c, 16'hb730, 16'h3171, 16'h3506}, // Filter 1 (was Input Channel 2)
            '{16'h338e, 16'h357d, 16'ha010, 16'h283b, 16'hacce, 16'hb1f4, 16'hbc7c, 16'hb8fe, 16'hb5b1}, // Filter 2 (was Input Channel 3)
            '{16'hb4e0, 16'h3102, 16'h346e, 16'h30c9, 16'h36fe, 16'haaad, 16'hb281, 16'hb563, 16'h32d8} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 1 Kernels (was Filter 2) ---
        '{
            '{16'hada6, 16'h2dd4, 16'hb04e, 16'h365f, 16'h320b, 16'h3347, 16'h2f62, 16'h2dab, 16'hae67}, // Filter 0 (was Input Channel 1)
            '{16'hb8cc, 16'h363e, 16'hb16d, 16'hb8dd, 16'haa0f, 16'h35c0, 16'ha9df, 16'ha93f, 16'h32a5}, // Filter 1 (was Input Channel 2)
            '{16'h2446, 16'hb172, 16'h3471, 16'hb431, 16'hb462, 16'h34fd, 16'hb58d, 16'hb696, 16'ha8f1}, // Filter 2 (was Input Channel 3)
            '{16'h3330, 16'hb283, 16'hb0b3, 16'h2bae, 16'haf92, 16'h9ea2, 16'h2b44, 16'hab76, 16'h33ed} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 2 Kernels (was Filter 3) ---
        '{
            '{16'h2a1a, 16'hb20d, 16'h3207, 16'h317d, 16'h2be4, 16'hb918, 16'h3685, 16'h33b6, 16'hb77d}, // Filter 0 (was Input Channel 1)
            '{16'hb158, 16'hb71b, 16'hb173, 16'hb042, 16'hb5bc, 16'hac0e, 16'hb4e8, 16'hb3d0, 16'h2924}, // Filter 1 (was Input Channel 2)
            '{16'ha9dc, 16'hb1bd, 16'h33c9, 16'h3477, 16'h254a, 16'ha370, 16'h30b5, 16'h3359, 16'h30c9}, // Filter 2 (was Input Channel 3)
            '{16'h3228, 16'hb820, 16'hb38a, 16'h317d, 16'hb378, 16'hbb25, 16'h383f, 16'hb13e, 16'hbac1} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 3 Kernels (was Filter 4) ---
        '{
            '{16'h2613, 16'had60, 16'h3045, 16'h2ff0, 16'h2f53, 16'h3120, 16'hb51d, 16'hb2d8, 16'h34e9}, // Filter 0 (was Input Channel 1)
            '{16'haef4, 16'h31bb, 16'h24f2, 16'h31d0, 16'h3111, 16'ha044, 16'h9634, 16'h34e6, 16'hb2e3}, // Filter 1 (was Input Channel 2)
            '{16'hb7c6, 16'hb793, 16'hbbc4, 16'hafe9, 16'hadf1, 16'hb889, 16'h3613, 16'h39ec, 16'h34e8}, // Filter 2 (was Input Channel 3)
            '{16'h3451, 16'hb43d, 16'hb4d0, 16'h34aa, 16'h274e, 16'hb641, 16'haa65, 16'hb8f7, 16'h36a9} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 4 Kernels (was Filter 5) ---
        '{
            '{16'h2f11, 16'h30d0, 16'hadab, 16'hb865, 16'hb827, 16'hb540, 16'hb1e5, 16'hb575, 16'hb4b3}, // Filter 0 (was Input Channel 1)
            '{16'hb596, 16'hb2c9, 16'hb5d8, 16'h3338, 16'h3140, 16'hb575, 16'h32de, 16'h31bc, 16'h30fa}, // Filter 1 (was Input Channel 2)
            '{16'had56, 16'hae76, 16'h326b, 16'hb638, 16'hb406, 16'hb381, 16'h3413, 16'h30f4, 16'ha83f}, // Filter 2 (was Input Channel 3)
            '{16'hb5b4, 16'h2730, 16'h3739, 16'haf85, 16'h341e, 16'h3789, 16'hb8a2, 16'hb317, 16'h2b4a} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 5 Kernels (was Filter 6) ---
        '{
            '{16'hb869, 16'hb651, 16'h31a2, 16'hb621, 16'hb96c, 16'h3088, 16'hb4d5, 16'h28c4, 16'h3724}, // Filter 0 (was Input Channel 1)
            '{16'h3266, 16'ha936, 16'hb362, 16'h3434, 16'hb0be, 16'hb6bc, 16'h2fb8, 16'hb6ca, 16'hb8b2}, // Filter 1 (was Input Channel 2)
            '{16'haf8a, 16'h35a5, 16'hb4ee, 16'habdf, 16'h30d2, 16'hba05, 16'h37f7, 16'h30fe, 16'hb214}, // Filter 2 (was Input Channel 3)
            '{16'hb3ef, 16'h302a, 16'haa51, 16'h3195, 16'ha782, 16'hb021, 16'h3170, 16'hb1e1, 16'had67} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 6 Kernels (was Filter 7) ---
        '{
            '{16'h307d, 16'ha6ca, 16'hb01b, 16'hb623, 16'h31cf, 16'h34cf, 16'h2e61, 16'hb256, 16'h2dae}, // Filter 0 (was Input Channel 1)
            '{16'h2f56, 16'h32dd, 16'h2d70, 16'h333b, 16'h3327, 16'h2cc1, 16'h381c, 16'hb1ac, 16'h2ad1}, // Filter 1 (was Input Channel 2)
            '{16'hb57f, 16'haec0, 16'h2fff, 16'h350c, 16'hb04c, 16'hb21a, 16'h32ad, 16'h2a83, 16'hb62c}, // Filter 2 (was Input Channel 3)
            '{16'hb1cd, 16'hb052, 16'haa96, 16'h3170, 16'hb440, 16'hb876, 16'hb74f, 16'h3346, 16'hb702} // Filter 3 (was Input Channel 4)
        },
        // --- Channel 7 Kernels (was Filter 8) ---
        '{
            '{16'hb7d5, 16'hac7f, 16'h2861, 16'hb87f, 16'h33b9, 16'h3031, 16'h316b, 16'h34b9, 16'h2210}, // Filter 0 (was Input Channel 1)
            '{16'h336b, 16'habab, 16'hb5f0, 16'h34ac, 16'hac15, 16'hb974, 16'hb32d, 16'hb63f, 16'hb428}, // Filter 1 (was Input Channel 2)
            '{16'ha82f, 16'hb1cf, 16'hb1d1, 16'ha660, 16'hab5b, 16'hb296, 16'hac34, 16'hb36f, 16'hb919}, // Filter 2 (was Input Channel 3)
            '{16'h2c83, 16'h200d, 16'h38cd, 16'hb537, 16'hb232, 16'h2ede, 16'hb02b, 16'h2abd, 16'h31cd} // Filter 3 (was Input Channel 4)
        }
    };

    // --- Biases for each of the 8 channels ---
    localparam [DATA_WIDTH-1:0] BIASES[0:NUM_CHANNELS-1] = '{16'hb06a, 16'haff3, 16'hb41b, 16'hb59b, 16'h28a8, 16'h3032, 16'h2e84, 16'hb438};


    //================================================================
    // Internal Signals
    //================================================================
    logic kernel_load_r;
    logic channel_valid_in;

    // --- State Machine for Kernel Loading ---
    typedef enum logic [1:0] {IDLE, LOAD, RUN} state_t;
    state_t state, next_state;
    logic [1:0] load_cycle_count;

    // --- Kernel wires to feed the channels ---
    logic [DATA_WIDTH-1:0] kernel_wires [0:NUM_CHANNELS-1][0:INPUT_CHANNEL_NUMBER-1][0:KERNEL_SIZE-1];
    logic [NUM_CHANNELS-1:0] valid_out_wires;

    //================================================================
    // Kernel Loading State Machine
    //================================================================
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
            IDLE:
                // After reset, move to the loading state
                next_state = LOAD;
            LOAD: begin
                kernel_load_r = 1'b1;
                // It takes 3 cycles to load the 3 columns of the kernels
                if (load_cycle_count == KERNEL_SIZE - 1) begin
                    next_state = RUN;
                end
            end
            RUN:
                // Stay in run state for normal operation
                next_state = RUN;
        endcase
    end

    // The valid signal to the channels is active during kernel loading OR normal operation
    assign channel_valid_in = valid_in | kernel_load_r;


    //================================================================
    // Kernel Muxing Logic
    //================================================================
    // Based on the load cycle, we select the correct kernel column to load.
    // When not loading, these inputs to the channel don't matter as `kernel_load` is low.
    always_comb begin
        for (int ch = 0; ch < NUM_CHANNELS; ch++) begin
            for (int f = 0; f < INPUT_CHANNEL_NUMBER; f++) begin
                for (int c = 0; c < KERNEL_SIZE; c++) begin
                    // KERNELS[ch][f][row*KERNEL_SIZE + col]
                    // We are loading column by column. The current column is load_cycle_count.
                    kernel_wires[ch][f][c] = KERNELS[ch][f][c*KERNEL_SIZE + load_cycle_count];
                end
            end
        end
    end


    //================================================================
    // Channel Instantiation
    //================================================================
    generate
        for (genvar ch_idx = 0; ch_idx < NUM_CHANNELS; ch_idx++) begin : gen_channel
            wire [DATA_WIDTH - 1:0] channel_output [INPUT_COL_SIZE - KERNEL_SIZE:0];
            wire [DATA_WIDTH - 1:0] ReLU_output [INPUT_COL_SIZE - KERNEL_SIZE:0];
            wire valid;
            cv3_channel #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE),
                .INPUT_COL_SIZE(INPUT_COL_SIZE),
                .INPUT_CHANNEL_NUMBER(INPUT_CHANNEL_NUMBER),
                .BIAS(BIASES[ch_idx])
            ) u_cv3_channel (
                .clk(clk),
                .rst(rst),
                .kernel_load(kernel_load_r),
                .valid_in(channel_valid_in),

                .input_columns(input_columns),
                .kernel_inputs(kernel_wires[ch_idx]),

                .output_column(channel_output),
                .valid_out(valid)
            );
            ReLU_column #(
                .COLUMN_SIZE(INPUT_COL_SIZE - KERNEL_SIZE + 1)
            ) ReLU (
                .data_in(channel_output),
                .data_out(ReLU_output)
            );
            pooling_layer #(
                .WINDOWS((INPUT_COL_SIZE - KERNEL_SIZE + 1)/2)
            ) Pooling (
                .clk(clk),
                .rst(rst),
                .valid_in(valid),
                .input_column(ReLU_output),
                .valid_out(valid_out_wires[ch_idx]),
                .output_column(output_columns[ch_idx])
            );

        end
    endgenerate

    assign valid_out = valid_out_wires[0];


endmodule
