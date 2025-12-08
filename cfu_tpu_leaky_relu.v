// Copyright 2021 The CFU-Playground Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`include "TPU.v"
`include "global_buffer_bram.v"
`include "leaky_relu.v"


module Cfu (
  input               cmd_valid,
  output reg             cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg            rsp_valid,
  input                 rsp_ready,
  output reg    [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);

wire rst_n = ~reset;

reg A_wr_en, B_wr_en, C_wr_en;
reg [31:0] A_data_in, B_data_in;
reg [127:0] C_data_in;
wire [31:0] A_data_out, B_data_out;
wire [127:0] C_data_out;
wire busy;
reg [11:0] CPU_A_idx, CPU_B_idx, CPU_C_idx;
reg [11:0] TPU_A_idx, TPU_B_idx, TPU_C_idx;      
reg [11:0] READ_A_idx, READ_B_idx, READ_C_idx;
reg [11:0] M_reg, N_reg, K_reg;                  
reg in_valid_reg;
reg [31:0] inputoffset_reg;

wire tpu_A_wr_en, tpu_B_wr_en, tpu_C_wr_en;
wire [11:0] tpu_A_index, tpu_B_index, tpu_C_index;
wire [31:0] tpu_A_data_in, tpu_B_data_in;
wire [127:0] tpu_C_data_in;        

reg [31:0] pos_multiplier_reg, neg_multiplier_reg;
reg [5:0] pos_shift_reg, neg_shift_reg;
reg [31:0] output_min_reg, output_max_reg;
reg [31:0] leaky_relu_input_reg, leaky_relu_inputoffset_reg, leaky_relu_outputoffset_reg;
wire [31:0] leaky_relu_result;


wire A_idx_out_of_bound = (TPU_A_idx >= (((N_reg + 3) >> 2) * K_reg)) ? 1'b1 : 1'b0;
wire B_idx_out_of_bound = (TPU_B_idx >= (((M_reg + 3) >> 2) * K_reg)) ? 1'b1 : 1'b0;

reg [31:0] tpu_A_data_out;
reg [31:0] tpu_B_data_out;

always @(*) begin
    if(A_idx_out_of_bound) begin
        tpu_A_data_out = 32'd0;
    end
    else begin
        tpu_A_data_out = A_data_out;
    end

    if(B_idx_out_of_bound) begin
        tpu_B_data_out = 32'd0;
    end
    else begin
        tpu_B_data_out = B_data_out;
    end
end


reg [31:0] is_A_idx_out_of_bound;
reg [31:0] is_B_idx_out_of_bound;
always @(posedge clk or posedge reset) begin
    if (reset) begin
        is_A_idx_out_of_bound <= 1'b0;
        is_B_idx_out_of_bound <= 1'b0;
    end
    else begin
        if(A_idx_out_of_bound)
            is_A_idx_out_of_bound <= {12'b0, TPU_A_idx, 8'b1};
        if(B_idx_out_of_bound)
            is_B_idx_out_of_bound <= {12'b0, TPU_B_idx, 8'b1};
    end
end

reg [6:0] State, Next_State;
localparam  IDLE       = 7'd0,
            LOAD_A     = 7'd1,
            LOAD_B     = 7'd2,
            LOAD_KMN   = 7'd3,
            LOAD_INPUTOFFSET = 7'd4,
            READ_A     = 7'd5,
            READ_B     = 7'd6,
            READ_C_word1     = 7'd7,
            READ_C_word2     = 7'd8,
            READ_C_word3     = 7'd9,
            READ_C_word4     = 7'd10,
            READ_KMN     = 7'd11,
            READ_INPUTOFFSET = 7'd12,
            COMPUTE_INIT  = 7'd13,
            COMPUTE_WAIT  = 7'd14,
            COMPUTE    = 7'd15,
            RESPOND    = 7'd16,
            READ_A_Idx_out_of_bound = 7'd17,
            READ_B_Idx_out_of_bound = 7'd18,
            LOAD_POS_MULTIPLIER = 7'd19,
            LOAD_NEG_MULTIPLIER = 7'd20,
            LOAD_POS_SHIFT = 7'd21,
            LOAD_NEG_SHIFT = 7'd22,
            LOAD_OUTPUT_MIN = 7'd23,
            LOAD_OUTPUT_MAX = 7'd24,
            LOAD_INPUT_DATA = 7'd25,
            LOAD_LEAKY_RELU_INPUTOFFSET = 7'd26,
            LOAD_LEAKY_RELU_OUTPUTOFFSET = 7'd27,
            READ_POS_MULTIPLIER = 7'd28,
            READ_NEG_MULTIPLIER = 7'd29,
            READ_POS_SHIFT = 7'd30,
            READ_NEG_SHIFT = 7'd31,
            READ_OUTPUT_MIN = 7'd32,
            READ_OUTPUT_MAX = 7'd33,
            READ_INPUT_DATA = 7'd34,
            READ_OUTPUT_DATA = 7'd35,
            READ_LEAKY_RELU_INPUTOFFSET = 7'd36,
            READ_LEAKY_RELU_OUTPUTOFFSET = 7'd37;

global_buffer_bram #(
    .ADDR_BITS(12),
    .DATA_BITS(32)
)
gbuff_A(
    .clk(clk),
    .rst_n(rst_n),
    .ram_en(1'b1),
    .wr_en(A_wr_en),
    .CPU_index(CPU_A_idx),
    .TPU_index(TPU_A_idx),
    .data_in(A_data_in),
    .data_out(A_data_out)
);

global_buffer_bram #(
    .ADDR_BITS(12),
    .DATA_BITS(32)
) gbuff_B(
    .clk(clk),
    .rst_n(rst_n),
    .ram_en(1'b1),
    .wr_en(B_wr_en),
    .CPU_index(CPU_B_idx),
    .TPU_index(TPU_B_idx),
    .data_in(B_data_in),
    .data_out(B_data_out)
);


global_buffer_bram #(
    .ADDR_BITS(12),
    .DATA_BITS(128)
) gbuff_C(
    .clk(clk),
    .rst_n(rst_n),
    .ram_en(1'b1),
    .wr_en(C_wr_en),
    .CPU_index(tpu_C_index[11:0]),
    .TPU_index(TPU_C_idx),
    .data_in(tpu_C_data_in),
    .data_out(C_data_out)
);


TPU tpu_inst (
    .clk(clk),
    .rst_n(rst_n),
    .in_valid(in_valid_reg),
    .K(K_reg),
    .M(M_reg),
    .N(N_reg),
    .busy(busy),
    .A_wr_en(tpu_A_wr_en),
    .A_index(tpu_A_index),
    .A_data_in(tpu_A_data_in),
    .A_data_out(tpu_A_data_out),
    .B_wr_en(tpu_B_wr_en),
    .B_index(tpu_B_index),
    .B_data_in(tpu_B_data_in),
    .B_data_out(tpu_B_data_out),
    .C_wr_en(tpu_C_wr_en),
    .C_index(tpu_C_index),
    .C_data_in(tpu_C_data_in),
    .C_data_out(C_data_out),
    .inputoffset(inputoffset_reg)
);

leaky_relu leaky_relu_inst (
    .x(leaky_relu_input_reg),
    .pos_multiplier(pos_multiplier_reg),
    .pos_shift(pos_shift_reg),
    .neg_multiplier(neg_multiplier_reg),
    .neg_shift(neg_shift_reg),
    .output_min(output_min_reg),
    .output_max(output_max_reg),
    .result(leaky_relu_result),
    .input_offset(leaky_relu_inputoffset_reg),
    .output_offset(leaky_relu_outputoffset_reg)
);


reg busy_reg;

always @(posedge clk or posedge reset) begin
    if (reset) begin
        busy_reg <= 1'b0;
    end
    else begin
        busy_reg <= busy;
    end
end

always @(*) begin
    if(busy || (State == IDLE && cmd_payload_function_id[9:3] == 10)) begin
        TPU_A_idx = tpu_A_index[11:0];
        TPU_B_idx = tpu_B_index[11:0];
        TPU_C_idx = tpu_C_index[11:0];
        C_wr_en = tpu_C_wr_en;
    end
    else begin
        TPU_A_idx = READ_A_idx;
        TPU_B_idx = READ_B_idx;
        TPU_C_idx = READ_C_idx;
        C_wr_en = 1'b0;
    end
    
    
end

reg [15:0] counter, next_counter;

always @(posedge clk or posedge reset) begin
    if (reset) begin
      State <= IDLE;
      counter <= 16'd0;
    end 
    else begin
      State <= Next_State;
      counter <= next_counter;
    end
end



always @(*) begin
    Next_State = State;
    next_counter = counter;
    case (State)
        IDLE: begin
            rsp_valid = 1'b0;
            if (cmd_valid) begin
                if (cmd_payload_function_id[9:3] == 0) begin
                    Next_State = LOAD_A;
                end
                else if (cmd_payload_function_id[9:3] == 1) begin
                    Next_State = LOAD_B;
                end
                else if (cmd_payload_function_id[9:3] == 2) begin
                    Next_State = LOAD_KMN;
                end
                else if (cmd_payload_function_id[9:3] == 3) begin
                    Next_State = READ_A;
                end
                else if (cmd_payload_function_id[9:3] == 4) begin
                    Next_State = READ_B;
                end
                else if (cmd_payload_function_id[9:3] == 5) begin
                    Next_State = READ_C_word1;
                end
                else if (cmd_payload_function_id[9:3] == 6) begin
                    Next_State = READ_C_word2;
                end
                else if (cmd_payload_function_id[9:3] == 7) begin
                    Next_State = READ_C_word3;
                end
                else if (cmd_payload_function_id[9:3] == 8) begin
                    Next_State = READ_C_word4;
                end
                else if (cmd_payload_function_id[9:3] == 9) begin
                    Next_State = READ_KMN;
                end
                else if (cmd_payload_function_id[9:3] == 10) begin
                    Next_State = COMPUTE_INIT;
                end
                else if (cmd_payload_function_id[9:3] == 11) begin
                    Next_State = LOAD_INPUTOFFSET;
                end
                else if (cmd_payload_function_id[9:3] == 12) begin
                    Next_State = READ_INPUTOFFSET;
                end
                else if (cmd_payload_function_id[9:3] == 13) begin
                    Next_State = READ_A_Idx_out_of_bound;
                end
                else if (cmd_payload_function_id[9:3] == 14) begin
                    Next_State = READ_B_Idx_out_of_bound;
                end
                else if (cmd_payload_function_id[9:3] == 19) begin
                    Next_State = LOAD_POS_MULTIPLIER;
                end
                else if (cmd_payload_function_id[9:3] == 20) begin
                    Next_State = LOAD_NEG_MULTIPLIER;
                end
                else if (cmd_payload_function_id[9:3] == 21) begin
                    Next_State = LOAD_POS_SHIFT;
                end
                else if (cmd_payload_function_id[9:3] == 22) begin
                    Next_State = LOAD_NEG_SHIFT;
                end
                else if (cmd_payload_function_id[9:3] == 23) begin
                    Next_State = LOAD_OUTPUT_MIN;
                end
                else if (cmd_payload_function_id[9:3] == 24) begin
                    Next_State = LOAD_OUTPUT_MAX;
                end
                else if (cmd_payload_function_id[9:3] == 25) begin
                    Next_State = LOAD_INPUT_DATA;
                end
                else if (cmd_payload_function_id[9:3] == 26) begin
                   Next_State = LOAD_LEAKY_RELU_INPUTOFFSET;
                end
                else if (cmd_payload_function_id[9:3] == 27) begin
                    Next_State = LOAD_LEAKY_RELU_OUTPUTOFFSET;
                end
                else if (cmd_payload_function_id[9:3] == 28) begin
                    Next_State = READ_POS_MULTIPLIER;
                end
                else if (cmd_payload_function_id[9:3] == 29) begin
                    Next_State = READ_NEG_MULTIPLIER;
                end
                else if (cmd_payload_function_id[9:3] == 30) begin
                    Next_State = READ_POS_SHIFT;
                end
                else if (cmd_payload_function_id[9:3] == 31) begin
                    Next_State = READ_NEG_SHIFT;
                end
                else if (cmd_payload_function_id[9:3] == 32) begin
                    Next_State = READ_OUTPUT_MIN;
                end
                else if (cmd_payload_function_id[9:3] == 33) begin
                    Next_State = READ_OUTPUT_MAX;
                end
                else if (cmd_payload_function_id[9:3] == 34) begin
                    Next_State = READ_INPUT_DATA;
                end
                else if (cmd_payload_function_id[9:3] == 35) begin
                    Next_State = READ_OUTPUT_DATA;
                end
                else if (cmd_payload_function_id[9:3] == 36) begin
                    Next_State = READ_LEAKY_RELU_INPUTOFFSET;
                end
                else if (cmd_payload_function_id[9:3] == 37) begin
                    Next_State = READ_LEAKY_RELU_OUTPUTOFFSET;
                end
            end
        end
        LOAD_A: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_B: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_KMN: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_INPUTOFFSET: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_POS_MULTIPLIER: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_NEG_MULTIPLIER: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_POS_SHIFT: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_NEG_SHIFT: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_OUTPUT_MIN: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_OUTPUT_MAX: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_INPUT_DATA: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_LEAKY_RELU_INPUTOFFSET: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        LOAD_LEAKY_RELU_OUTPUTOFFSET: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end
        READ_A: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_A;
            end
        end
        READ_B: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_B;
            end
        end
        READ_C_word1: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_C_word1;
            end
        end
        READ_C_word2: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_C_word2;
            end
        end
        READ_C_word3: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_C_word3;
            end
        end
        READ_C_word4: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin  
                Next_State = READ_C_word4;
            end
        end
        READ_KMN: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_KMN;
            end
        end
        READ_INPUTOFFSET: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_INPUTOFFSET;
            end
        end
        READ_A_Idx_out_of_bound: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_A_Idx_out_of_bound;
            end
        end
        READ_B_Idx_out_of_bound: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_B_Idx_out_of_bound;
            end
        end
        READ_POS_MULTIPLIER: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_POS_MULTIPLIER;
            end
        end
        READ_NEG_MULTIPLIER: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_NEG_MULTIPLIER;
            end
        end
        READ_POS_SHIFT: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_POS_SHIFT;
            end
        end
        READ_NEG_SHIFT: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_NEG_SHIFT;
            end
        end
        READ_OUTPUT_MIN: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_OUTPUT_MIN;
            end
        end
        READ_OUTPUT_MAX: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_OUTPUT_MAX;
            end
        end
        READ_INPUT_DATA: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_INPUT_DATA;
            end
        end
        READ_OUTPUT_DATA: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_OUTPUT_DATA;
            end
        end
        READ_LEAKY_RELU_INPUTOFFSET: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_LEAKY_RELU_INPUTOFFSET;
            end
        end
        READ_LEAKY_RELU_OUTPUTOFFSET: begin
            rsp_valid = 1'b0;
            if(rsp_ready) begin
                Next_State = RESPOND;
            end
            else begin
                Next_State = READ_LEAKY_RELU_OUTPUTOFFSET;
            end
        end
        COMPUTE_INIT: begin
            rsp_valid = 1'b0;
            Next_State = COMPUTE_WAIT;
        end
        COMPUTE_WAIT: begin
            rsp_valid = 1'b0;
            Next_State = COMPUTE;
        end
        COMPUTE: begin
            rsp_valid = 1'b0;
            if(busy) begin
                Next_State = COMPUTE;
            end
            else begin
                Next_State = RESPOND;
            end
        end
        RESPOND: begin
            rsp_valid = 1'b1;
            Next_State = IDLE;
        end

        default: begin
            rsp_valid = 1'b0;
            Next_State = IDLE;
        end
    endcase
end

always @(*) begin
    cmd_ready = State == IDLE;
end

always @(posedge clk or posedge reset) begin
    if (reset) begin
        A_wr_en <= 1'b0;
        B_wr_en <= 1'b0;
        in_valid_reg <= 1'b0;
        K_reg <= 12'd0;
        M_reg <= 12'd0;
        N_reg <= 12'd0;
        rsp_payload_outputs_0 <= 32'd0;
        CPU_A_idx <= 12'd0;
        CPU_B_idx <= 12'd0;
        A_data_in <= 32'd0;
        B_data_in <= 32'd0;
        inputoffset_reg <= 32'd0;
        pos_multiplier_reg <= 32'd0;
        neg_multiplier_reg <= 32'd0;
        pos_shift_reg <= 6'd0;
        neg_shift_reg <= 6'd0;
        output_min_reg <= 32'd0;
        output_max_reg <= 32'd0;
        leaky_relu_input_reg <= 32'd0;
        leaky_relu_inputoffset_reg <= 32'd0;
        leaky_relu_outputoffset_reg <= 32'd0;
    end
    else begin
        in_valid_reg <= 1'b0;
        A_wr_en <= 1'b0;
        B_wr_en <= 1'b0;
        
        case(State)
            IDLE: begin
                if (cmd_valid) begin
                    if (cmd_payload_function_id[9:3] == 0) begin
                        A_wr_en <= 1'b1;
                        CPU_A_idx <= cmd_payload_inputs_1[11:0];
                        A_data_in <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 1) begin
                        B_wr_en <= 1'b1;
                        CPU_B_idx <= cmd_payload_inputs_1[11:0];
                        B_data_in <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 2) begin
                        K_reg <= cmd_payload_inputs_0[11:0];
                        M_reg <= cmd_payload_inputs_1[23:12];
                        N_reg <= cmd_payload_inputs_1[11:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 3) begin
                        READ_A_idx <= cmd_payload_inputs_1[11:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 4) begin
                        READ_B_idx <= cmd_payload_inputs_1[11:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 5 || cmd_payload_function_id[9:3] == 6 || cmd_payload_function_id[9:3] == 7 || cmd_payload_function_id[9:3] == 8) begin
                        READ_C_idx <= cmd_payload_inputs_1[11:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 11) begin
                        inputoffset_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 19) begin
                        pos_multiplier_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 20) begin
                        neg_multiplier_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 21) begin
                        pos_shift_reg <= cmd_payload_inputs_0[5:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 22) begin
                        neg_shift_reg <= cmd_payload_inputs_0[5:0];
                    end
                    else if (cmd_payload_function_id[9:3] == 23) begin
                        output_min_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 24) begin
                        output_max_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 25) begin
                        leaky_relu_input_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 26) begin
                        leaky_relu_inputoffset_reg <= cmd_payload_inputs_0;
                    end
                    else if (cmd_payload_function_id[9:3] == 27) begin
                        leaky_relu_outputoffset_reg <= cmd_payload_inputs_0;
                    end
                end
            end
            READ_A: begin
                rsp_payload_outputs_0 <= A_data_out;
            end
            READ_B: begin
                rsp_payload_outputs_0 <= B_data_out;
            end
            READ_C_word1: begin
                rsp_payload_outputs_0 <= C_data_out[127:96];
            end
            READ_C_word2: begin
                rsp_payload_outputs_0 <= C_data_out[95:64];
            end
            READ_C_word3: begin
                rsp_payload_outputs_0 <= C_data_out[63:32];
            end
            READ_C_word4: begin
                rsp_payload_outputs_0 <= C_data_out[31:0];
            end
            READ_KMN: begin
                rsp_payload_outputs_0 <= {K_reg[7:0], M_reg, N_reg};
            end
            READ_INPUTOFFSET: begin
                rsp_payload_outputs_0 <= inputoffset_reg;
            end
            READ_A_Idx_out_of_bound: begin
                rsp_payload_outputs_0 <= is_A_idx_out_of_bound;
            end
            READ_B_Idx_out_of_bound: begin
                rsp_payload_outputs_0 <= is_B_idx_out_of_bound;
            end
            READ_POS_MULTIPLIER: begin
                rsp_payload_outputs_0 <= pos_multiplier_reg;
            end
            READ_NEG_MULTIPLIER: begin
                rsp_payload_outputs_0 <= neg_multiplier_reg;
            end
            READ_POS_SHIFT: begin
                rsp_payload_outputs_0 <= {26'd0, pos_shift_reg};
            end
            READ_NEG_SHIFT: begin
                rsp_payload_outputs_0 <= {{26{neg_shift_reg[5]}}, neg_shift_reg};
            end
            READ_OUTPUT_MIN: begin
                rsp_payload_outputs_0 <= output_min_reg;
            end
            READ_OUTPUT_MAX: begin
                rsp_payload_outputs_0 <= output_max_reg;
            end
            READ_INPUT_DATA: begin  
                rsp_payload_outputs_0 <= leaky_relu_input_reg;
            end
            READ_OUTPUT_DATA: begin  
                rsp_payload_outputs_0 <= leaky_relu_result;
            end
            READ_LEAKY_RELU_INPUTOFFSET: begin
                rsp_payload_outputs_0 <= leaky_relu_inputoffset_reg;
            end
            READ_LEAKY_RELU_OUTPUTOFFSET: begin
                rsp_payload_outputs_0 <= leaky_relu_outputoffset_reg;
            end
            COMPUTE_INIT: begin
                in_valid_reg <= 1'b1;
            end
            COMPUTE_WAIT: begin
                in_valid_reg <= 1'b0;
            end
            COMPUTE: begin
                in_valid_reg <= 1'b0;
            end
        endcase  
    end
end

endmodule
