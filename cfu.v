module Cfu (
    input               cmd_valid,
    output              cmd_ready,
    input      [9:0]    cmd_payload_function_id,
    input      [31:0]   cmd_payload_inputs_0,
    input      [31:0]   cmd_payload_inputs_1,
    output reg          rsp_valid,
    input               rsp_ready,
    output reg [31:0]   rsp_payload_outputs_0,
    input               reset,
    input               clk
  );

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

  // Only not ready for a command when we have a response.
  assign cmd_ready = ~rsp_valid;



  always @(posedge clk)
  begin
    if (reset)
    begin
      InputOffset <= 9'd128;
      FilterOffset <= 9'd0;
      rsp_payload_outputs_0 <= 32'b0;
      rsp_valid <= 1'b0;
    end
    else if (rsp_valid)
    begin
      rsp_valid <= ~rsp_ready;
    end
    else if (cmd_valid)
    begin
      rsp_valid <= 1'b1;

      if (cmd_payload_function_id[9:3] == 0)
      begin
        rsp_payload_outputs_0 <= rsp_payload_outputs_0 + sum_prods;
      end
      else if (cmd_payload_function_id[9:3] == 1)
      begin
        rsp_payload_outputs_0 <= 32'b0;
      end
      else if (cmd_payload_function_id[9:3] == 2)
      begin
        InputOffset <= cmd_payload_inputs_0[8:0];
      end
      else if(cmd_payload_function_id[9:3] == 3)
      begin
        FilterOffset <= cmd_payload_inputs_0[8:0];
      end
    end
  end
endmodule