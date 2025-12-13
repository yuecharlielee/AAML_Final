`include "systolic_array.v"
`include "buffer.v"
`include "PE.v"

module TPU(
    clk,
    rst_n,
    in_valid,
    K,
    M,
    N,
    busy,
    A_wr_en,
    A_index,
    A_data_in,
    A_data_out,
    B_wr_en,
    B_index,
    B_data_in,
    B_data_out,
    C_wr_en,
    C_index,
    C_data_in,
    C_data_out,
    inputoffset
);

input clk;
input rst_n;
input            in_valid;
input [15:0]      K;
input [15:0]      M;
input [15:0]      N;
output reg       busy;

output reg          A_wr_en;
output reg [15:0]   A_index;
output reg [31:0]   A_data_in;
input  [31:0]       A_data_out;

output reg          B_wr_en;
output reg [15:0]   B_index;
output reg [31:0]   B_data_in;
input  [31:0]       B_data_out;

(* mark_debug = "true" *) output reg          C_wr_en;
(* mark_debug = "true" *) output reg [15:0]   C_index;
(* mark_debug = "true" *) output reg [127:0]  C_data_in;
input  [127:0]      C_data_out;

input [31:0]       inputoffset;

reg [31:0] A_buffer_in, B_buffer_in;
wire [31:0] A_buffer_out, B_buffer_out;

wire [127:0] c_out[3:0];

buffer weight_buffer_A (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .data_in(A_buffer_in),
    .data_out(A_buffer_out)
);

buffer input_buffer_B (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid),
    .data_in(B_buffer_in),
    .data_out(B_buffer_out)
);
reg sa_valid;

systolic_array SA (
    .rst_n(rst_n),
    .in_valid(sa_valid),
    .clk(clk),
    .in_a(A_buffer_out),
    .in_b(B_buffer_out),
    .out_c_r1(c_out[0]),
    .out_c_r2(c_out[1]),
    .out_c_r3(c_out[2]),
    .out_c_r4(c_out[3]),
    .inputoffset(inputoffset)
);

reg [15:0] M_reg, N_reg, K_reg;
reg [15:0] tile_m, tile_n, tile_k;
reg [15:0] max_tile_m, max_tile_n, max_tile_k;

(* mark_debug = "true" *) wire [127:0] c_row0 = c_out[0];
(* mark_debug = "true" *) wire [127:0] c_row1 = c_out[1];
(* mark_debug = "true" *) wire [127:0] c_row2 = c_out[2];
(* mark_debug = "true" *) wire [127:0] c_row3 = c_out[3];

integer i, j;

localparam IDLE = 3'd0;
localparam LOAD = 3'd1;
localparam CALC = 3'd2;
localparam WRITEBACK = 3'd3;
localparam DONE = 3'd4;

reg [3:0] state, next_state;
reg [15:0] load_cnt, calc_cnt, wb_cnt;

// wire [15:0] A_addr = tile_k * 4 * max_tile_m + tile_m * 4 + load_cnt;
// wire [15:0] B_addr = tile_k * 4 * max_tile_n + tile_n * 4 + load_cnt;  
// wire [15:0] C_addr = tile_n * M_reg + tile_m * 4 + wb_cnt;

reg [15:0] A_addr, B_addr, C_addr;



always @(*) begin
    if(tile_k == max_tile_k - 1) begin
        if(tile_k * 4 + load_cnt > K_reg) begin
            A_addr = (((tile_m * K_reg) + K_reg)) ;
            B_addr = (((tile_n * K_reg) + K_reg));
        end
        else begin
            A_addr = (((tile_m * K_reg) + (tile_k * 4))) + load_cnt;
            B_addr = (((tile_n * K_reg) + (tile_k * 4))) + load_cnt;
        end
    end
    else begin
        A_addr = (((tile_m * K_reg) + (tile_k * 4))) + load_cnt;
        B_addr = (((tile_n * K_reg) + (tile_k * 4))) + load_cnt;
    end


    C_addr = ((tile_n * M_reg) + (tile_m * 4)) + wb_cnt;

    // if(tile_k == max_tile_k - 1) begin
    //     if(tile_k * 4 + load_cnt > K_reg) begin
    //         C_addr = ((tile_n * K_reg) + K_reg) + wb_cnt;
    //     end
    //     else begin
    //         C_addr = (((tile_n * K_reg) + (tile_k * 4))) + wb_cnt;
    //     end
    // end
    // else begin
    //     C_addr = (((tile_n * K_reg) + (tile_k * 4))) + wb_cnt;
    // end
end

reg [15:0] next_tile_m, next_tile_n;
always @(*) begin
    if(tile_n < max_tile_n - 1) begin
        next_tile_n = tile_n + 1;
        next_tile_m = tile_m;
    end
    else begin
        next_tile_n = 0;
        if(tile_m < max_tile_m - 1) begin
            next_tile_m = tile_m + 1;
        end
        else begin
            next_tile_m = 0;
        end
    end
end

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        M_reg <= 0;
        N_reg <= 0;
        K_reg <= 0;
        max_tile_m <= 0;
        max_tile_n <= 0; 
        max_tile_k <= 0;
    end
    else if(in_valid) begin
        M_reg <= M;
        N_reg <= N;
        K_reg <= K;
        max_tile_m <= (M + 3) >> 2;  
        max_tile_n <= (N + 3) >> 2; 
        max_tile_k <= (K + 3) >> 2; 
    end
end

always @(*) begin
    case(state)
        IDLE: begin
            if(in_valid)
                next_state = LOAD;
            else
                next_state = IDLE;
        end
        LOAD: begin
            if(load_cnt > K_reg || tile_k * 4 + load_cnt > K_reg) 
                next_state = CALC;
            else
                next_state = LOAD;
        end
        CALC: begin
            if(calc_cnt == 7)
                next_state = WRITEBACK;    
            else
                next_state = CALC;
        end
        WRITEBACK: begin
            if(wb_cnt > 3 || tile_m * 4 + wb_cnt >= M_reg) begin
                if(tile_n < max_tile_n - 1 || tile_m < max_tile_m - 1)
                    next_state = LOAD;
                else
                    next_state = DONE;
            end
            else
                next_state = WRITEBACK;
        end
        DONE: begin
            next_state = IDLE;
        end
        default: next_state = IDLE;
    endcase
end

always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        busy <= 0;
        A_wr_en <= 0;
        B_wr_en <= 0;
        C_wr_en <= 0;
        A_index <= 0;
        B_index <= 0;
        C_index <= 0;
        load_cnt <= 0;
        calc_cnt <= 0;
        wb_cnt <= 0;
        sa_valid <= 0;
        state <= IDLE;
        tile_k <= 0;
        tile_m <= 0;
        tile_n <= 0;
    end
    else begin
        state <= next_state;
        
        case(state)
            IDLE: begin
                if(in_valid) begin
                    tile_k <= 0;
                    tile_m <= 0;
                    tile_n <= 0;
                    busy <= 1;
                    load_cnt <= 0;
                    calc_cnt <= 0;
                    wb_cnt <= 0;
                    sa_valid <= 1;
                end
                else begin
                    busy <= 0;
                end
                A_buffer_in <= 0;
                B_buffer_in <= 0;
                C_wr_en <= 0;
            end
            
            LOAD: begin
                A_index <= A_addr;
                B_index <= B_addr;
                sa_valid <= 0;
                
                if(tile_k * 4 + load_cnt > K_reg) begin
                    A_buffer_in <= 0; 
                    B_buffer_in <= 0;
                end
                else if(load_cnt == 0) begin
                    A_buffer_in <= 0; 
                    B_buffer_in <= 0;
                end
                else begin
                    A_buffer_in <= A_data_out;
                    B_buffer_in <= B_data_out;
                end
                
                if(load_cnt > K_reg) begin
                    load_cnt <= 0;
                end
                else begin
                    load_cnt <= load_cnt + 1;
                end
            end
            
            CALC: begin
                A_buffer_in <= 0;
                B_buffer_in <= 0;
                
                if(calc_cnt == 7) begin
                    calc_cnt <= 0;
                end
                else begin
                    calc_cnt <= calc_cnt + 1;
                end
            end
            
            WRITEBACK: begin
                if(wb_cnt <= 3 && tile_m * 4 + wb_cnt < M_reg) begin
                    C_wr_en <= 1;
                    C_index <= C_addr;
                    C_data_in <= c_out[wb_cnt];
                    wb_cnt <= wb_cnt + 1;
                end
                else begin
                    C_wr_en <= 0;
                    sa_valid <= 1;
                    tile_m <= next_tile_m;
                    tile_n <= next_tile_n;
                    wb_cnt <= 0;
                end
            end
            DONE: begin
                busy <= 0;
                C_wr_en <= 0;
            end
        endcase
    end
end


endmodule