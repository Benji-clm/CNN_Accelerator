module conv_layer_2 #(
    parameter DATA_WIDTH           = 16,
    parameter KERNEL_SIZE          = 4,
    parameter INPUT_COL_SIZE       = 5,
    parameter NUM_CHANNELS         = 10,
    parameter INPUT_CHANNEL_NUMBER = 8 //For testing purpose, halve the input channel number
)(
    input logic clk,
    input logic rst,

    // --- Control Signals ---
    input logic valid_in,

    // --- Data Inputs ---
    input logic [DATA_WIDTH-1:0] input_columns [INPUT_CHANNEL_NUMBER-1:0][INPUT_COL_SIZE-1:0],


    // --- Data Outputs ---
    // Output size updated to reflect no pooling layer.
    output logic [DATA_WIDTH-1:0] output_columns[NUM_CHANNELS-1:0][(INPUT_COL_SIZE - KERNEL_SIZE + 1) - 1:0],
    output logic valid_out
);

    // Kernels are now 10 (NUM_CHANNELS) x 8 (INPUT_CHANNEL_NUMBER) x 16 (KERNEL_SIZE*KERNEL_SIZE)
    // Values have been converted from float to 16-bit hex format.
    localparam [15:0] KERNELS[0:9][0:7][0:15] = '{
        // Filter 0
        '{
            {16'h2752, 16'hB090, 16'hB2F0, 16'hB4BA, 16'h34F9, 16'hAA15, 16'hAAB3, 16'hBA1E, 16'h2AE8, 16'hB1A3, 16'hB46A, 16'hB5F7, 16'h2F3A, 16'hA82D, 16'hAF35, 16'hABE4},
            {16'hAC8B, 16'hB10B, 16'hAF7A, 16'h38A7, 16'hB32D, 16'hB1BF, 16'h3024, 16'h37B8, 16'h2FDE, 16'hA4EA, 16'h28A0, 16'hAEBF, 16'h3568, 16'h319C, 16'h3625, 16'hBBEE},
            {16'hABFD, 16'hA4F9, 16'hB3D9, 16'hB8C2, 16'h2FF4, 16'h2220, 16'hA722, 16'h3145, 16'hB08F, 16'h2F6B, 16'h321D, 16'h2E89, 16'hAF42, 16'h2EC1, 16'hB531, 16'hAA4E},
            {16'h2FA1, 16'hB109, 16'hB466, 16'h32F1, 16'hB371, 16'hB4F8, 16'hACF0, 16'h34AA, 16'hB827, 16'hAA16, 16'h29DE, 16'h2EA3, 16'hB4A4, 16'hB7EB, 16'hB452, 16'hB878},
            {16'hB4FF, 16'hB0C8, 16'hB09F, 16'hB5BF, 16'hB652, 16'hAE27, 16'hAEB6, 16'hB3D5, 16'hBAEF, 16'hB716, 16'hB38B, 16'hA6F7, 16'hACE0, 16'hAA70, 16'h2DFF, 16'h3552},
            {16'hB0B9, 16'hB1C9, 16'h2226, 16'h309B, 16'h2AFB, 16'hA23A, 16'hB0EC, 16'hA846, 16'h2EC5, 16'h3335, 16'hAF5D, 16'h2647, 16'h3608, 16'hA963, 16'hBBA2, 16'hB778},
            {16'h3613, 16'h3311, 16'h31BB, 16'h2FAB, 16'hAA77, 16'h2DD9, 16'hB460, 16'hB838, 16'hABEC, 16'hAF78, 16'hB03E, 16'hAACE, 16'h2A02, 16'h291C, 16'hA243, 16'hA2FC},
            {16'h2CA1, 16'h3281, 16'hAE8A, 16'hB6D1, 16'h3401, 16'h2B71, 16'hAC55, 16'hA801, 16'h2A02, 16'hB285, 16'hB154, 16'h2E02, 16'h30D5, 16'hB80E, 16'hB77E, 16'hAACA}
        },
        // Filter 1
        '{
            {16'h2A2F, 16'hB2FF, 16'hB806, 16'hB47B, 16'h2940, 16'hB664, 16'hBA59, 16'hB64D, 16'hB379, 16'hB40A, 16'hB7E4, 16'hB4DA, 16'hADCE, 16'hB53B, 16'hB1CB, 16'h267F},
            {16'hA80E, 16'h3811, 16'h36BB, 16'h3129, 16'hB86A, 16'hACA0, 16'h2A28, 16'hA6D4, 16'hB750, 16'hB406, 16'hA60B, 16'h31AC, 16'hB19D, 16'hB68C, 16'hA565, 16'h3700},
            {16'hAA7D, 16'h317D, 16'h3333, 16'h2C75, 16'h30E4, 16'h2C30, 16'h321A, 16'hAF8C, 16'hAEE7, 16'hA868, 16'h306A, 16'h1651, 16'hB175, 16'h32D1, 16'h2E85, 16'hB095},
            {16'hB4B9, 16'hB704, 16'hB7BF, 16'hB684, 16'hA19B, 16'hB448, 16'hB20E, 16'hB29A, 16'hACB7, 16'hB578, 16'hAF10, 16'hB46B, 16'hB5CA, 16'hB836, 16'hB84C, 16'hB41D},
            {16'hB363, 16'h20A2, 16'h20EE, 16'hB658, 16'hA9B0, 16'hB2EE, 16'hB7E5, 16'hB07E, 16'h2E21, 16'hB452, 16'hB8FA, 16'h2757, 16'h1CA4, 16'hB193, 16'h359F, 16'h319E},
            {16'hAB9B, 16'h2A87, 16'h3444, 16'h3601, 16'hB415, 16'hAE18, 16'h2FB0, 16'h2B45, 16'hAD2A, 16'h2CDA, 16'h3559, 16'hB676, 16'h2E41, 16'hA988, 16'hAA7A, 16'h38A9},
            {16'hB538, 16'hAB85, 16'hA936, 16'hAD36, 16'h29B6, 16'hB088, 16'hB2C1, 16'hB18F, 16'h33C2, 16'hA5BD, 16'h98DB, 16'h3253, 16'h332D, 16'h3527, 16'h34E8, 16'h34C1},
            {16'hB7CB, 16'hAC76, 16'h30C1, 16'h34EE, 16'hB9F0, 16'hAC1B, 16'h295B, 16'h3293, 16'hB481, 16'h2CBF, 16'hA559, 16'hAF6A, 16'h3423, 16'h344D, 16'hA63D, 16'hB95D}
        },
        // Filter 2
        '{
            {16'hB131, 16'hAE77, 16'hB14D, 16'hB95E, 16'hADEA, 16'hAC32, 16'hAD79, 16'hB3E8, 16'h2D87, 16'h9539, 16'h9ACC, 16'h2CF7, 16'h3450, 16'h3316, 16'h3240, 16'h2FC3},
            {16'hB7A8, 16'hB05B, 16'h2E56, 16'h2FB7, 16'hB82B, 16'hB4D5, 16'hAD5F, 16'h25E3, 16'hB2CE, 16'h13B3, 16'hAE71, 16'hB3D4, 16'hB112, 16'h1415, 16'h3492, 16'h35CB},
            {16'hA599, 16'h274F, 16'h3143, 16'h2405, 16'hB412, 16'hB0AC, 16'h2CED, 16'h30AF, 16'h34A0, 16'h2C6E, 16'h31EE, 16'hA96A, 16'h38C1, 16'hB0F9, 16'hB84A, 16'hB903},
            {16'h3516, 16'h3429, 16'h3481, 16'h3468, 16'h2E5F, 16'h9CFA, 16'h2CC2, 16'hA4A1, 16'hB0D6, 16'h2837, 16'h3111, 16'hB696, 16'hB68D, 16'hB356, 16'hAC55, 16'hB258},
            {16'hA4BB, 16'h2D4C, 16'h294F, 16'hB83B, 16'h2DDE, 16'h2605, 16'h2C4D, 16'hB188, 16'h22FF, 16'hA34E, 16'hADEE, 16'h3108, 16'h2433, 16'h2B4D, 16'hB137, 16'h37F1},
            {16'h1E9D, 16'hAA8B, 16'hA6E2, 16'hB176, 16'hB577, 16'hB2E7, 16'hA430, 16'h3147, 16'h3191, 16'h2FD5, 16'h2D0F, 16'h2CE8, 16'hA425, 16'hAB92, 16'hAFEC, 16'hAB40},
            {16'h2D30, 16'hAFCA, 16'hAC9F, 16'hB6CA, 16'h3855, 16'h35D2, 16'h2EC5, 16'h2655, 16'h34B3, 16'h347B, 16'h2D4B, 16'h2018, 16'hAB96, 16'h34BF, 16'hAB4E, 16'hB5B6},
            {16'h35C1, 16'hB171, 16'hB6DA, 16'hB1CB, 16'hB350, 16'hB10C, 16'hB09C, 16'hB150, 16'h2A24, 16'h81D6, 16'h3091, 16'h30FA, 16'h2F18, 16'hB46C, 16'hB2B9, 16'hA909}
        },
        // Filter 3
        '{
            {16'hB5C1, 16'hB1EB, 16'hAE9E, 16'h341B, 16'hB376, 16'h2992, 16'h2BF9, 16'h354F, 16'hB3C0, 16'h2FA0, 16'h307C, 16'h2E40, 16'hB639, 16'hAE15, 16'hABBF, 16'h1CAF},
            {16'hB255, 16'hADA5, 16'hB08B, 16'hA81C, 16'hB946, 16'hB463, 16'h2E08, 16'h2297, 16'hB078, 16'hB287, 16'h9867, 16'hAD9F, 16'h34EE, 16'h31FF, 16'h318C, 16'hB01F},
            {16'hBA01, 16'hB979, 16'h36E6, 16'h35DD, 16'hB79E, 16'hB1B9, 16'hB2B1, 16'hAA97, 16'hB449, 16'h2B0E, 16'hA262, 16'h333B, 16'hB424, 16'hB0C1, 16'h342C, 16'h3596},
            {16'h37A9, 16'h35DF, 16'h3168, 16'h3434, 16'hA1B2, 16'h3205, 16'h28D2, 16'h9D11, 16'hB2C8, 16'hB181, 16'hAD35, 16'h3516, 16'hB272, 16'h2F49, 16'h316A, 16'h36AC},
            {16'h2F81, 16'h22B7, 16'h314F, 16'hB86D, 16'hA4AA, 16'h1B7A, 16'h3479, 16'hAEBE, 16'h292E, 16'h3200, 16'h3435, 16'hB467, 16'hA927, 16'h330D, 16'h2DEA, 16'hB689},
            {16'hB34B, 16'hB2C0, 16'hB02A, 16'hB012, 16'hB29B, 16'hB24D, 16'hB469, 16'hB504, 16'h98EF, 16'hB287, 16'hB28E, 16'hB57D, 16'hB376, 16'hB714, 16'hB40C, 16'hB6FD},
            {16'h2356, 16'h3232, 16'h3019, 16'hAF13, 16'hB720, 16'hAFAD, 16'h2DEC, 16'h3210, 16'hB1E3, 16'hB03D, 16'hAAA8, 16'hB3D0, 16'h31FE, 16'h2CBF, 16'h323B, 16'h2E31},
            {16'hB52B, 16'hAF15, 16'h2F9A, 16'h385F, 16'hB334, 16'hB055, 16'hADA7, 16'h2C89, 16'hB7DD, 16'hA93B, 16'hB083, 16'hABB7, 16'hBA72, 16'hB6A3, 16'hB1D9, 16'hB28C}
        },
        // Filter 4
        '{
            {16'hB24F, 16'h2551, 16'h2CC2, 16'hAC24, 16'h1BE6, 16'hAE24, 16'h1B00, 16'h2702, 16'h33EF, 16'h29CF, 16'h2E50, 16'h224E, 16'h2F00, 16'hB4FC, 16'hB54E, 16'hAB03},
            {16'h385B, 16'h327B, 16'h2CF8, 16'h2F3E, 16'h3222, 16'h313D, 16'h2DE3, 16'h3179, 16'h24BA, 16'h31A9, 16'h2C87, 16'hA80E, 16'hAFE7, 16'hB252, 16'hB61C, 16'hB37D},
            {16'h31B1, 16'h355C, 16'h34D0, 16'h31C8, 16'h2DBD, 16'h32A1, 16'h2CE3, 16'h1813, 16'hA47E, 16'hB2E4, 16'hB2A7, 16'hB27A, 16'hB099, 16'hAB9D, 16'hB560, 16'hB551},
            {16'h29E1, 16'hB475, 16'hB541, 16'hB86D, 16'hA854, 16'h2DFA, 16'h9FBC, 16'hB6B6, 16'h301A, 16'h3144, 16'hA3AD, 16'hB70D, 16'h35FD, 16'h342F, 16'h2AF7, 16'hB1DD},
            {16'hB4B8, 16'hB49D, 16'hB50F, 16'hB2D2, 16'hB699, 16'h24C8, 16'hB2B9, 16'hB488, 16'h3587, 16'hA816, 16'h2928, 16'hA850, 16'h300D, 16'hB25E, 16'h2AFD, 16'hB313},
            {16'h3476, 16'h3555, 16'h3417, 16'h34D3, 16'hB451, 16'hAF2A, 16'hB3C7, 16'h314B, 16'h1E12, 16'hB063, 16'hB46B, 16'h3128, 16'hB41C, 16'h338C, 16'h364E, 16'h36AD},
            {16'hAF01, 16'hB23D, 16'hAFBE, 16'hB316, 16'h35EF, 16'h2BE6, 16'h9AF3, 16'hB25C, 16'h08EA, 16'hB7C3, 16'hB071, 16'h33D8, 16'hB748, 16'hB714, 16'hB502, 16'hB8CF},
            {16'hB74E, 16'hB4E8, 16'hA19D, 16'h2F23, 16'hAE50, 16'hB6AA, 16'hAF4B, 16'h3449, 16'h3126, 16'hA705, 16'h298D, 16'h35CC, 16'h30B9, 16'h32B0, 16'h347F, 16'h35E3}
        },
        // Filter 5
        '{
            {16'h2DC0, 16'h32D4, 16'h2F71, 16'h377A, 16'hAD62, 16'h2AAA, 16'hAF05, 16'h2C20, 16'hB183, 16'h222A, 16'hAE82, 16'hB9CD, 16'hB4E7, 16'hB08D, 16'hB43F, 16'hB48F},
            {16'hAFF5, 16'h2B2C, 16'h8BDD, 16'hB2AB, 16'h2B31, 16'h2CBA, 16'hB41E, 16'hB1A6, 16'h3214, 16'h2B35, 16'hB0DE, 16'h31C5, 16'h2F86, 16'h3151, 16'hA226, 16'hAA55},
            {16'hACDD, 16'hB1FD, 16'hB9A8, 16'hB1D9, 16'hAE8A, 16'h2DF9, 16'hB6F8, 16'hB84D, 16'hB619, 16'hA4A0, 16'h31B2, 16'h300E, 16'hB84A, 16'h3448, 16'h3523, 16'h3516},
            {16'h9EE7, 16'hB5FA, 16'hB93E, 16'hBCCE, 16'h3278, 16'hB088, 16'hA74C, 16'hB03A, 16'hA0A8, 16'h2949, 16'h2AB6, 16'h3324, 16'h2B98, 16'h294B, 16'h336F, 16'h350B},
            {16'hB806, 16'hAD30, 16'h21EF, 16'h3800, 16'h240B, 16'h320B, 16'hACC7, 16'h369E, 16'h3258, 16'h2BE4, 16'hAFFA, 16'h3591, 16'h340C, 16'hA3BC, 16'hACAE, 16'hB631},
            {16'h253C, 16'h336A, 16'hAA3C, 16'h2EC8, 16'hAA33, 16'h33A3, 16'h2C39, 16'hB483, 16'hA652, 16'h3307, 16'h2141, 16'hAB6F, 16'hADBE, 16'hB87D, 16'hB8CE, 16'hAD2A},
            {16'hB552, 16'h9D98, 16'hA5BA, 16'h30C0, 16'hACE6, 16'hB2A2, 16'h2E3D, 16'hAFA0, 16'h2CE9, 16'hB1AD, 16'h2807, 16'h22ED, 16'hB0D0, 16'h3558, 16'h346A, 16'h2C85},
            {16'hA91A, 16'h2C14, 16'h2D76, 16'hB148, 16'hAFE8, 16'h3100, 16'h2819, 16'hB30F, 16'hB3E3, 16'h2429, 16'hA5BC, 16'hB0FC, 16'hB893, 16'hB2A5, 16'hB56A, 16'hB523}
        },
        // Filter 6
        '{
            {16'hB144, 16'h2E83, 16'hA484, 16'hACAF, 16'hB0CB, 16'h3284, 16'h2A90, 16'h2687, 16'hA922, 16'h22A0, 16'h30F5, 16'h2909, 16'hAA19, 16'h27DD, 16'h3100, 16'h26C2},
            {16'h3498, 16'h2F9E, 16'hB0A8, 16'hBA11, 16'h3347, 16'hB0FB, 16'hB430, 16'hB1C1, 16'h30F4, 16'hA9A9, 16'h3052, 16'h30C9, 16'h2E6A, 16'h31CD, 16'h339F, 16'hA402},
            {16'h3540, 16'h3817, 16'h3626, 16'hABC5, 16'h3476, 16'hB321, 16'hB72B, 16'hB669, 16'h2827, 16'h2E2D, 16'hAA3D, 16'h3189, 16'hB53E, 16'hB69E, 16'hB469, 16'h30A3},
            {16'hAC0D, 16'hB6D2, 16'hB328, 16'hB822, 16'hB72B, 16'hB81D, 16'h3072, 16'hADFE, 16'hB8B1, 16'hB17F, 16'hB022, 16'h3164, 16'hB5B5, 16'hB402, 16'hAC81, 16'h2C6A},
            {16'hB862, 16'hB8F9, 16'hB689, 16'h2892, 16'hBA2E, 16'hB8E1, 16'hB1DE, 16'hA2D4, 16'hB9A6, 16'hB855, 16'hB0C7, 16'hAEF5, 16'hB06E, 16'h327D, 16'h3114, 16'h268D},
            {16'h2FEF, 16'h2E60, 16'h3413, 16'hB65E, 16'h2C62, 16'hAD65, 16'h29AD, 16'hB6F1, 16'h2F73, 16'h310B, 16'hB22B, 16'hB56F, 16'h292F, 16'hAF0C, 16'hBC4A, 16'hBB2C},
            {16'h2A56, 16'hB51A, 16'hB54D, 16'hAE8F, 16'h3806, 16'h306C, 16'h328C, 16'h3865, 16'hAA71, 16'h31AC, 16'hA632, 16'h8DA7, 16'hAE3B, 16'hACAD, 16'hAFE2, 16'hAC7F},
            {16'h34A8, 16'h3448, 16'h3542, 16'hA581, 16'hA8A0, 16'hAA4C, 16'h24F8, 16'hB7E4, 16'h31AB, 16'h21EE, 16'hAAC1, 16'hB507, 16'h305F, 16'hB34E, 16'hB5B7, 16'hAC6D}
        },
        // Filter 7
        '{
            {16'hB6F6, 16'hB8A7, 16'hB981, 16'hA945, 16'hB5A4, 16'hB622, 16'hB7B0, 16'hA988, 16'hB265, 16'hB5B8, 16'hB240, 16'h3093, 16'hB7B0, 16'hB3D8, 16'hB3B6, 16'h2D41},
            {16'h384B, 16'h3546, 16'h30C1, 16'hB6E8, 16'h2B4E, 16'hB513, 16'hA9A9, 16'h256E, 16'hB620, 16'hB25A, 16'h2EA9, 16'h3072, 16'hBABB, 16'hB7B5, 16'hB657, 16'hB760},
            {16'hAECB, 16'hBA75, 16'hB97B, 16'h3284, 16'hA1AA, 16'h32F5, 16'h30B9, 16'h2998, 16'hB4A6, 16'h345D, 16'hB090, 16'hB5AE, 16'h25F2, 16'h2FCC, 16'h3133, 16'h21B0},
            {16'hB4AD, 16'h3307, 16'h2F95, 16'hA59B, 16'hB03A, 16'h2EDC, 16'h31E0, 16'h31CE, 16'h3404, 16'hB1C4, 16'hB14D, 16'hA1F0, 16'h35D5, 16'h2338, 16'hB359, 16'hBA39},
            {16'h3103, 16'h2EBC, 16'h293D, 16'h3480, 16'h31FA, 16'h9C48, 16'hB202, 16'hB026, 16'h31E0, 16'hADBB, 16'hAD7E, 16'h325F, 16'h2637, 16'hB272, 16'hB38F, 16'hB4AA},
            {16'hAF0E, 16'h2E29, 16'hA8E4, 16'hB159, 16'h3465, 16'h332F, 16'h2D5C, 16'h2F4A, 16'hB03D, 16'hB509, 16'hB461, 16'h334B, 16'hB1A3, 16'h33E5, 16'h36A6, 16'h35D9},
            {16'h381A, 16'hA99F, 16'hA870, 16'h33C7, 16'h30FC, 16'h3008, 16'hB3C2, 16'hB534, 16'hB097, 16'h3483, 16'h3226, 16'h3564, 16'hB481, 16'hB91A, 16'hB863, 16'hB9A2},
            {16'hAFCE, 16'hB0E0, 16'hB3FC, 16'hB11F, 16'h356A, 16'h3071, 16'h2C31, 16'h31A6, 16'h3171, 16'hB0EE, 16'hAF22, 16'hA448, 16'h9829, 16'h360E, 16'h3716, 16'h34EB}
        },
        // Filter 8
        '{
            {16'h342F, 16'h3012, 16'h3286, 16'h2CF2, 16'h2A9C, 16'hAC66, 16'h1838, 16'h364A, 16'h3112, 16'h2DA0, 16'hA61E, 16'h3526, 16'h333F, 16'h3286, 16'h349E, 16'h2B3E},
            {16'hB7EF, 16'hB048, 16'h2D9A, 16'h2DFF, 16'h34CB, 16'h3684, 16'h31F5, 16'hA98A, 16'hAFDA, 16'h2966, 16'h2B43, 16'hB209, 16'hAC8E, 16'hB034, 16'h3035, 16'h2EAB},
            {16'hAE26, 16'hB429, 16'hB05D, 16'h2AF1, 16'hB312, 16'hB765, 16'hB357, 16'hAE43, 16'h3378, 16'hB024, 16'hACDB, 16'hB419, 16'hA39B, 16'hB787, 16'h3138, 16'h363C},
            {16'hB691, 16'hAF48, 16'hAC82, 16'h3206, 16'h389A, 16'h3354, 16'h207F, 16'h2A95, 16'h3588, 16'hA9C2, 16'hAB36, 16'hA0D5, 16'h346E, 16'hAE75, 16'hA721, 16'h356F},
            {16'hB180, 16'hA5DC, 16'hAF78, 16'hAF61, 16'hB422, 16'hB24B, 16'hAD43, 16'hB249, 16'hB211, 16'hB505, 16'h2C88, 16'h2820, 16'hB602, 16'hB2B3, 16'hACA1, 16'hACB7},
            {16'h2C78, 16'hAD10, 16'hACDD, 16'hB113, 16'hADDB, 16'hA78D, 16'h1EDA, 16'hB3F2, 16'hB191, 16'hB016, 16'h186C, 16'hB47C, 16'h2D55, 16'hA45C, 16'hB516, 16'hB805},
            {16'hB47E, 16'h2F1D, 16'h2F3C, 16'h277C, 16'hB3F7, 16'hAC77, 16'hA73B, 16'hA4BE, 16'h24AF, 16'h302A, 16'h2706, 16'hB661, 16'hB35A, 16'h2968, 16'hB192, 16'hB1F2},
            {16'h1511, 16'hB338, 16'hAEB5, 16'h3123, 16'hB538, 16'hACCC, 16'hACBF, 16'hAFC1, 16'h29D3, 16'h313D, 16'hAB3F, 16'hB262, 16'h2F77, 16'hB130, 16'hB1C5, 16'hB7DF}
        },
        // Filter 9
        '{
            {16'h36AB, 16'h34A1, 16'h349A, 16'hB23B, 16'h3015, 16'h286D, 16'h2D95, 16'hB5C2, 16'hA4AF, 16'h2B03, 16'hAC19, 16'hB87C, 16'hB465, 16'hB452, 16'hB42C, 16'hB641},
            {16'hB705, 16'hB2A4, 16'hAA44, 16'h3171, 16'h31BA, 16'hA4C1, 16'h28F3, 16'h31A0, 16'h36D7, 16'h3082, 16'hB05C, 16'hB083, 16'hA9B0, 16'hADF6, 16'hB6D2, 16'h2F34},
            {16'hA7E2, 16'hB48A, 16'hB75F, 16'hB22A, 16'hAD7C, 16'hB47D, 16'h3503, 16'h350C, 16'h31E3, 16'hBA9B, 16'hA084, 16'h2B85, 16'hAE6F, 16'hB319, 16'hACD2, 16'hB2B4},
            {16'hB7FA, 16'hA792, 16'h1875, 16'h2D73, 16'hBA86, 16'h2E64, 16'hAD49, 16'h91E0, 16'hAE26, 16'h3027, 16'hA9C2, 16'hB559, 16'h2995, 16'h2D35, 16'h2A50, 16'hB5A0},
            {16'hB7FA, 16'hB3A8, 16'h2816, 16'hB4FD, 16'hB52D, 16'h2BE0, 16'h2340, 16'hB3C9, 16'hB344, 16'h33D5, 16'h33B0, 16'hB59F, 16'h30BB, 16'h2F7D, 16'hB05F, 16'hB8E7},
            {16'hB19C, 16'hB8F8, 16'hB77C, 16'hB202, 16'h34E4, 16'h2F98, 16'hA3F2, 16'hB10C, 16'h2FD0, 16'h31ED, 16'h2E28, 16'h2A0E, 16'hAEBC, 16'h2FF2, 16'h322E, 16'h3893},
            {16'hA3DE, 16'h3099, 16'h32C3, 16'h3633, 16'hB4FE, 16'hAB84, 16'hB0A5, 16'h29B6, 16'hBB19, 16'hB54A, 16'hA94F, 16'h27FA, 16'h245E, 16'hB2FC, 16'hB7A1, 16'hAB26},
            {16'h2F99, 16'h3123, 16'hA738, 16'h2736, 16'hAB3A, 16'hAA52, 16'hADBB, 16'hADCD, 16'hB587, 16'hB1B0, 16'hAE1C, 16'hAE02, 16'hA890, 16'h2498, 16'h34B3, 16'h3142}
        }
    };

    localparam [15:0] BIASES[0:9] = '{16'h34F1, 16'h393A, 16'hAD8D, 16'hB4ED, 16'hB65F, 16'h2E6D, 16'hB3A6, 16'h3120, 16'hB010, 16'hB45A};


    //================================================================
    // Internal Signals
    //================================================================
    logic kernel_load_r;
    logic channel_valid_in;

    // --- State Machine for Kernel Loading ---
    typedef enum logic [1:0] {IDLE, LOAD, RUN} state_t;
    state_t state, next_state;
    // Counter needs to count up to KERNEL_SIZE-1. For KERNEL_SIZE=4, we need 2 bits.
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
                // It takes KERNEL_SIZE cycles to load the KERNEL_SIZE columns of the kernels
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
            // Instantiating the new cv4_channel module
            cv4_channel #(
                .DATA_WIDTH(DATA_WIDTH),
                .KERNEL_SIZE(KERNEL_SIZE),
                .INPUT_COL_SIZE(INPUT_COL_SIZE),
                .INPUT_CHANNEL_NUMBER(INPUT_CHANNEL_NUMBER),
                .BIAS(BIASES[ch_idx])
            ) u_cv4_channel (
                .clk(clk),
                .rst(rst),
                .kernel_load(kernel_load_r),
                .valid_in(channel_valid_in),

                .input_columns(input_columns),
                .kernel_inputs(kernel_wires[ch_idx]),

                // Connect output directly, no ReLU or pooling
                .output_column(output_columns[ch_idx]),
                .valid_out(valid_out_wires[ch_idx])
            );

        end
    endgenerate

    // All valid_out signals from the channels should be synchronous.
    // We can assign one of them to the final output valid signal.
    assign valid_out = valid_out_wires[0];


endmodule
