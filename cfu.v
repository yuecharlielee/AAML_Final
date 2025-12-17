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

localparam IDLE = 2'b00;
localparam PROCESS = 2'b01;
localparam RESPOND = 2'b10;


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
            NextState = RESPOND;
        end
        RESPOND: begin
            if (rsp_ready)
                NextState = IDLE;
        end
    endcase
end

always @(posedge clk) begin
    if (reset) begin
        rsp_valid <= 0;
    end
    else if(State == IDLE) begin
        rsp_valid <= 0;
    end
    else if (State == PROCESS) begin
        rsp_valid <= 0;
    end
    else if(State == RESPOND) begin
        rsp_valid <= 1;
    end
end

endmodule
