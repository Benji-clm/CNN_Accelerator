module addfp16 (
    input  logic [15:0] a,
    input  logic [15:0] b,
    output logic [15:0] sum
);

    // Unpacked inputs
    logic a_sign, b_sign;
    logic [4:0] a_exp, b_exp;
    logic [9:0] a_man, b_man;

    // Intermediate signals
    logic [5:0]  expDiff;
    logic [10:0] manDiff;
    logic expDiffIsZero;
    logic manDiffIsZero;
    logic bigSel; // If 1, b is bigger. If 0, a is bigger/equal to b
    logic [4:0] rawShiftRt;
    logic [3:0] shiftRtAmt;
    logic operation; // 1 means subtraction, 0 means addition
    logic [4:0] rawExp;
    logic signOut;
    logic [10:0] bigMan;
    logic [10:0] lilMan;
    logic [10:0] shiftedMan;
    logic guard;
    logic [10:0] mask;
    logic sticky;
    logic [12:0] alignedMan;
    logic [13:0] rawMan;
    logic [13:0] signedMan;
    logic [3:0] rawNormAmt;
    logic valid;
    logic [3:0] normAmt;
    logic [5:0] biasExp;
    logic [14:0] biasMan;
    logic [5:0] normExp;
    logic [4:0] expOut;
    logic [14:0] normMan;
    logic [9:0] manOut;
    logic [14:0] numOut;

    // Exception signals
    logic expAIsOne, expBIsOne;
    logic expAIsZero, expBIsZero;
    logic manAIsZero, manBIsZero;
    logic AIsNaN, BIsNaN;
    logic AIsInf, BIsInf;
    logic inIsNaN;
    logic inIsInf;
    logic inIsDenorm;
    logic zero;
    logic expOutIsOne;
    logic expOutIsZero;
    logic outIsInf;
    logic outIsDenorm;
    logic overflow;
    logic NaN;
    logic underflow;


    // Unpack inputs
    assign a_sign = a[15];
    assign a_exp  = a[14:10];
    assign a_man  = a[9:0];

    assign b_sign = b[15];
    assign b_exp  = b[14:10];
    assign b_man  = b[9:0];

    // Compare Exponents
    assign expDiff = {1'b0, a_exp} - {1'b0, b_exp};
    assign manDiff = {1'b0, a_man} - {1'b0, b_man};

    assign expDiffIsZero = ~|expDiff;
    assign manDiffIsZero = ~|manDiff;

    assign bigSel = expDiffIsZero ? manDiff[10] : expDiff[5];

    assign rawShiftRt = expDiff[5] ? -expDiff[4:0] : expDiff[4:0];
    assign shiftRtAmt = (rawShiftRt > 11) ? 4'b1011 : rawShiftRt[3:0];

    // Determine operation (Add/Subtract)
    assign operation = a_sign ^ b_sign;

    // Select larger and smaller operands
    assign rawExp  = bigSel ? b_exp : a_exp;
    assign signOut = bigSel ? b_sign : a_sign;
    assign bigMan  = {1'b1, bigSel ? b_man : a_man};
    assign lilMan  = {1'b1, bigSel ? a_man : b_man};


    // Align mantissas
    assign shiftedMan = lilMan >> shiftRtAmt;
    assign guard = ((shiftRtAmt == 0) || (rawShiftRt > 11)) ? 1'b0 : lilMan[shiftRtAmt-1];

    // Create mask to determine sticky bit
    // This logic replaces the CASEZ to remove the overlap warning
    always_comb begin
        if (rawShiftRt > 11) begin
            mask = 11'h7FF; // If shift is large, all lower bits could be shifted
        end else begin
            case (shiftRtAmt)
                4'd1:  mask = 11'h001;
                4'd2:  mask = 11'h003;
                4'd3:  mask = 11'h007;
                4'd4:  mask = 11'h00F;
                4'd5:  mask = 11'h01F;
                4'd6:  mask = 11'h03F;
                4'd7:  mask = 11'h07F;
                4'd8:  mask = 11'h0FF;
                4'd9:  mask = 11'h1FF;
                4'd10: mask = 11'h3FF;
                4'd11: mask = 11'h7FF;
                default: mask = 11'b0; // Covers shift by 0
            endcase
        end
    end

    assign sticky = |(lilMan[10:0] & mask);

    assign alignedMan = operation ? ~{shiftedMan, guard, sticky} : {shiftedMan, guard, sticky};

    // Add Mantissas
    assign rawMan = alignedMan + {bigMan, 2'b0} + operation;
    assign signedMan = {~operation & rawMan[13], rawMan[12:0]};

    // Normalize
    lzd14 lzd(signedMan, rawNormAmt, valid);

    assign normAmt = valid ? rawNormAmt : 4'b0;

    assign biasExp = rawExp + 1;
    assign biasMan = {1'b0, signedMan};

    assign normExp = biasExp - {2'b0, normAmt};
    assign expOut  = normExp[4:0];

    assign normMan = biasMan << normAmt;
    assign manOut  = normMan[12:3];

    assign numOut = {expOut, manOut};

    // --- CORRECTED EXCEPTION LOGIC ---
    // Input exception signals
    assign expAIsOne  = &a_exp;
    assign expBIsOne  = &b_exp;
    assign expAIsZero = ~|a_exp;
    assign expBIsZero = ~|b_exp;
    assign manAIsZero = ~|a_man;
    assign manBIsZero = ~|b_man;

    assign AIsNaN = expAIsOne & ~manAIsZero;
    assign BIsNaN = expBIsOne & ~manBIsZero;
    assign AIsInf = expAIsOne & manAIsZero;
    assign BIsInf = expBIsOne & manAIsZero;

    assign inIsNaN    = AIsNaN | BIsNaN | (AIsInf & BIsInf & operation);
    assign inIsInf    = AIsInf | BIsInf;
    assign inIsDenorm = (expAIsZero & ~manAIsZero) | (expBIsZero & ~manBIsZero);
    assign zero       = expDiffIsZero & manDiffIsZero & operation;

    // Output exception signals
    assign expOutIsOne  = &expOut;
    assign expOutIsZero = ~|expOut;
    
    // Corrected logic based on the reference Verilog 
    assign outIsInf     = expOutIsOne & ~normExp[5];
    // Corrected logic based on the reference Verilog 
    assign outIsDenorm  = expOutIsZero & ~outIsInf & ~zero;

    // Final exception signals
    // Corrected logic based on the reference Verilog 
    assign overflow  = (inIsInf | outIsInf) & ~zero;
    assign NaN       = inIsNaN;
    // Corrected logic based on the reference Verilog 
    assign underflow = inIsDenorm | outIsDenorm | zero;

    // Choose output from exception signals
    assign sum = NaN       ? 16'b0_11111_1111111111 :
                 overflow  ? {signOut, 5'b11111, 10'b0000000000} :
                 underflow ? 16'b0 :
                             {signOut, numOut};

endmodule
