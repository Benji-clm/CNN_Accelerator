module reduction #(
    parameter data_width = 16,
    parameter mat_height = 2
)(
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         valid_in,
    input  logic [data_width-1:0]        column [mat_height-1:0],
    output logic                         valid_out,
    output logic [data_width-1:0]        sum
);

typedef enum logic [1:0] {
    IDLE,
    LOAD_2,
    OUTPUT
} state_t;

state_t current_state, next_state;

logic [data_width-1:0] val_1, val_2, val_3, val_4;
logic valid_out_reg;

always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
        current_state   <= IDLE;
        val_1           <= 0;
        val_2           <= 0;
        val_3           <= 0;
        val_4           <= 0;
        valid_out_reg   <= 0;
    end else begin
        current_state <= next_state;

        case (current_state)
            IDLE: begin
                if (valid_in) begin
                    val_1 <= column[0];
                    val_2 <= column[1];
                end
            end

            LOAD_2: begin
                val_3 <= column[0];
                val_4 <= column[1];
                valid_out_reg <= 0;
            end

            OUTPUT: begin
                valid_out_reg <= 1;
            end

            default: begin
                valid_out_reg <= 0;
            end
        endcase
    end
end

always_comb begin
    next_state = current_state;
    case (current_state)
        IDLE:    if (valid_in) next_state = LOAD_2;
        LOAD_2:  next_state = OUTPUT;
        OUTPUT:  next_state = IDLE;
    endcase
end

assign sum = val_1 + val_2 + val_3 + val_4; // combinational
assign valid_out = valid_out_reg;           // registered

endmodule
