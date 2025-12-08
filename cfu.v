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

`include "leaky_relu.v"

module Cfu (
  input               cmd_valid,
  output reg          cmd_ready,
  input      [9:0]    cmd_payload_function_id,
  input      [31:0]   cmd_payload_inputs_0,
  input      [31:0]   cmd_payload_inputs_1,
  output reg          rsp_valid,
  input               rsp_ready,
  output reg [31:0]   rsp_payload_outputs_0,
  input               reset,
  input               clk
);



reg [1:0] State, NextState;

// SIMD multiply step:
reg signed [8:0] InputOffset;
reg signed [8:0] FilterOffset;
wire signed [15:0] prod_0, prod_1, prod_2, prod_3;
  assign prod_0 =  ($signed(cmd_payload_inputs_0[7 : 0]) + InputOffset)
         * ($signed(cmd_payload_inputs_1[7 : 0]) + FilterOffset);
  assign prod_1 =  ($signed(cmd_payload_inputs_0[15: 8]) + InputOffset)
         * ($signed(cmd_payload_inputs_1[15: 8]) + FilterOffset);
  assign prod_2 =  ($signed(cmd_payload_inputs_0[23:16]) + InputOffset)
         * ($signed(cmd_payload_inputs_1[23:16]) + FilterOffset);
  assign prod_3 =  ($signed(cmd_payload_inputs_0[31:24]) + InputOffset)
         * ($signed(cmd_payload_inputs_1[31:24]) + FilterOffset);

wire signed [31:0] sum_prods;
assign sum_prods = (prod_0 + prod_1) + (prod_2 + prod_3);

// LeakyReLU parameters storage
reg signed [31:0] pos_multiplier_reg;  // output_multiplier_identity
reg signed [31:0] neg_multiplier_reg;  // output_multiplier_alpha
reg signed [31:0] pos_shift_reg;       // output_shift_identity
reg signed [31:0] neg_shift_reg;       // output_shift_alpha
reg signed [31:0] output_min_reg;
reg signed [31:0] output_max_reg;
reg signed [31:0] leaky_relu_input_reg, leaky_relu_inputoffset_reg, leaky_relu_outputoffset_reg;
wire signed [31:0] leaky_relu_result;

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

localparam IDLE = 2'b00;
localparam PROCESS = 2'b01;
localparam RESPOND = 2'b10;

// Only not ready for a command when we have a response.
always @(*) begin
    cmd_ready = (State == IDLE);
end

reg [31:0] result_reg;


always @(posedge clk) begin
if (reset)
    State <= IDLE;
else
    State <= NextState;
end

always @(*) begin
    NextState = State;
    case (State)
        IDLE: begin
            if (cmd_valid)
                NextState = PROCESS;
        end
        PROCESS: begin
            if(rsp_ready)
                NextState = IDLE;
        end
    endcase
end

always @(posedge clk) begin
    if (reset) begin
        InputOffset <= 9'd128;
        FilterOffset <= 9'd0;
        rsp_payload_outputs_0 <= 32'b0;
        rsp_valid <= 0;
        result_reg <= 32'b0;
        pos_multiplier_reg <= 32'b0;
        neg_multiplier_reg <= 32'b0;
        pos_shift_reg <= 32'b0;
        neg_shift_reg <= 32'b0;
        output_min_reg <= 32'b0;
        output_max_reg <= 32'b0;
    end
    else if (rsp_valid) begin
        rsp_valid <= ~rsp_ready;
    end
    else if (cmd_valid) begin
        rsp_valid <= 1;
        case (cmd_payload_function_id[9:3])
            0: rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
            1: begin
                rsp_payload_outputs_0 <= 32'b0;
            end
            2: InputOffset <= cmd_payload_inputs_0[8:0];
            3: FilterOffset <= cmd_payload_inputs_0[8:0];
            19: pos_multiplier_reg <= cmd_payload_inputs_0;   // LOAD_POS_MULTIPLIER
            20: neg_multiplier_reg <= cmd_payload_inputs_0;   // LOAD_NEG_MULTIPLIER
            21: pos_shift_reg <= cmd_payload_inputs_0;        // LOAD_POS_SHIFT
            22: neg_shift_reg <= cmd_payload_inputs_0;        // LOAD_NEG_SHIFT
            23: output_min_reg <= cmd_payload_inputs_0;       // LOAD_OUTPUT_MIN
            24: output_max_reg <= cmd_payload_inputs_0;       // LOAD_OUTPUT_MAX
            25: leaky_relu_input_reg <= cmd_payload_inputs_0;    // LOAD_LEAKY_RELU_INPUT
            26: leaky_relu_inputoffset_reg <= cmd_payload_inputs_0;   // LOAD_LEAKY_RELU_INPUTOFFSET
            27: leaky_relu_outputoffset_reg <= cmd_payload_inputs_0;  // LOAD_LEAKY_RELU_OUTPUTOFFSET
            35: rsp_payload_outputs_0 <= leaky_relu_result;               // COMPUTE_LEAKY_RELU
        endcase
    end
end

endmodule
