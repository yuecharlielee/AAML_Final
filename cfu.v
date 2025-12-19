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

`include "global_buffer_bram.v"


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

wire rst_n = ~reset;

localparam ADDR_BITS = 14;
localparam DATA_BITS = 8;
localparam Tile_size = 32;

reg [5:0] State, NextState;

localparam IDLE = 6'd0,
           PROCESS = 6'd1,
           RESPOND = 6'd2;

reg [Tile_size-1:0] wr_en_input;
reg [ADDR_BITS-1:0] index_input;
reg [DATA_BITS-1:0] data_in_input;
wire [DATA_BITS-1:0] data_out_input [31:0];

genvar i;
generate
    for (i = 0; i < 32; i = i + 1) begin : gbuff_instances
        global_buffer_bram #(
            .ADDR_BITS(ADDR_BITS),
            .DATA_BITS(DATA_BITS)
        ) gbuff_input(
            .clk(clk),
            .rst_n(rst_n),
            .ram_en(1'b1),
            .wr_en(wr_en_input[i]),
            .index(index_input),
            .data_in(data_in_input),
            .data_out(data_out_input[i])
        );
    end
endgenerate




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
            if (cmd_valid) begin
                NextState = PROCESS;
            end
        end
        PROCESS: begin
            NextState = RESPOND;
        end
        RESPOND: begin
            if (rsp_ready)
                NextState = IDLE;
        end
    endcase
end

always @(*) begin
    if(State == PROCESS) begin
        if(cmd_payload_function_id[9:3] == 0) begin
            wr_en_input = 1'b1 << cmd_payload_inputs_1[31:16];
            index_input = cmd_payload_inputs_1[15:0];
            data_in_input = cmd_payload_inputs_0[7:0];
        end
        else if(cmd_payload_function_id[9:3] == 1) begin
            wr_en_input = 0;
            index_input = cmd_payload_inputs_1[15:0];
            data_in_input = 0;
        end
        else begin
            wr_en_input = 0;
            index_input = 0;
            data_in_input = 0;
        end
    end
    else if(State == RESPOND) begin
        wr_en_input = 0;
        data_in_input = 0;
        if(cmd_payload_function_id[9:3] == 1) begin
            index_input = cmd_payload_inputs_1[15:0];
        end
        else begin
            index_input = 0;
        end
        
    end
    else begin
        wr_en_input = 0;
        index_input = 0;
        data_in_input = 0;
    end
end

always @(posedge clk) begin
    if (reset) begin
        rsp_payload_outputs_0 <= 0;
        rsp_valid <= 0;
    end
    else if(State == IDLE) begin
        rsp_payload_outputs_0 <= 0;
        rsp_valid <= 0;
    end
    else if (State == PROCESS) begin
        rsp_payload_outputs_0 <= 0;
        rsp_valid <= 0;
    end
    else if(State == RESPOND) begin
        rsp_valid <= 1;

        if(cmd_payload_function_id[9:3] == 0) begin
            rsp_payload_outputs_0 <= 0;
        end
        else if(cmd_payload_function_id[9:3] == 1) begin
            rsp_payload_outputs_0 <= {24'b0, data_out_input[cmd_payload_inputs_0[15:0]][7:0]};
        end
        else begin
            rsp_payload_outputs_0 <= 0;
        end
    end
end

endmodule