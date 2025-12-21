module global_buffer_bram #(parameter ADDR_BITS=14, parameter DATA_BITS=8)(
    input                      clk,
    input                      rst_n,
    input                      ram_en,
    input                      wr_en,
    input      [ADDR_BITS-1:0] index,
    input      [DATA_BITS-1:0] data_in,
    output reg [DATA_BITS-1:0] data_out
  );
  parameter DEPTH = 2**ADDR_BITS;
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (posedge clk)
  begin
    if (ram_en)
    begin
      if(wr_en)
      begin
        gbuff[index] <= data_in;
      end
      data_out <= gbuff[index];
    end
  end
endmodule

module result_buffer_lut #(parameter ADDR_BITS=5, parameter DATA_BITS=32)(
    input                      clk,
    input                      rst_n,
    input                      ram_en,
    input                      wr_en,
    input      [ADDR_BITS-1:0] index,
    input      [DATA_BITS-1:0] data_in,
    output reg [DATA_BITS-1:0] data_out
  );
  parameter DEPTH = 2**ADDR_BITS;
  (* ram_style = "distributed" *)
  reg [DATA_BITS-1:0] gbuff [DEPTH-1:0];

  always @ (posedge clk)
  begin
    if (ram_en)
    begin
      if(wr_en)
      begin
        gbuff[index] <= data_in;
      end
      data_out <= gbuff[index];
    end
  end
endmodule

module leaky_relu(
    input              clk,
    input              reset,
    input              start,
    input  wire signed [31:0]  x,

    input  wire signed [31:0]  pos_multiplier,
    input  wire signed [ 5:0]  pos_shift,
    input  wire signed [31:0]  neg_multiplier,
    input  wire signed [ 5:0]  neg_shift,

    input  wire signed [31:0]  input_offset,
    input  wire signed [31:0]  output_offset,
    input  wire signed [31:0]  output_min,
    input  wire signed [31:0]  output_max,

    output reg  signed [31:0]  result
  );
  wire signed [31:0] x_adjusted = x - input_offset;
  wire signed [31:0] selected_mult = (x_adjusted[31]) ? neg_multiplier : pos_multiplier;
  wire signed [ 5:0] selected_shift_wire = (x_adjusted[31]) ? neg_shift : pos_shift;

  reg signed [63:0] ab_64_reg;
  reg signed [ 5:0] shift_reg;

  reg signed [31:0] s1_x_shifted;
  reg signed [ 5:0] s1_right_shift;

  always @(posedge clk)
  begin
    if (start)
    begin
      if ($signed(selected_shift_wire) > 0)
      begin
        s1_x_shifted   = x_adjusted <<< selected_shift_wire;
        s1_right_shift = 6'd0;
      end
      else
      begin
        s1_x_shifted   = x_adjusted;
        s1_right_shift = -$signed(selected_shift_wire);
      end

      ab_64_reg <= $signed(s1_x_shifted) * $signed(selected_mult);
      shift_reg <= s1_right_shift;
    end
  end

  reg signed [63:0] acc_64;
  reg signed [31:0] srdhm_result;
  reg signed [31:0] raw_shifted;
  reg signed [31:0] unclamped_output;

  always @(*)
  begin
    acc_64 = ab_64_reg + 64'h00000000_3FFFFFFF;
    srdhm_result = acc_64[62:31];
    raw_shifted = srdhm_result >>> shift_reg;
    unclamped_output = raw_shifted + output_offset;

    if (unclamped_output > output_max)
      result = output_max;
    else if (unclamped_output < output_min)
      result = output_min;
    else
      result = unclamped_output;
  end
endmodule

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

  localparam TILE_SIZE = 32;

  localparam RESULT_ADDR_BITS = $clog2(TILE_SIZE);
  localparam ADDR_BITS = 14;
  localparam DATA_BITS = 8;
  localparam RES_DATA_BITS = 32;

  reg [5:0] State, NextState;
  localparam IDLE         = 6'd0,
             WRITE_INPUT  = 6'd1,
             PROCESS      = 6'd2,
             COMPUTE_MUL  = 6'd3,
             COMPUTE_ADD  = 6'd4,
             WRITE_BACK   = 6'd5,
             RESPOND      = 6'd6;

  reg [TILE_SIZE-1:0] wr_en_input;
  reg [ADDR_BITS-1:0] index_input;
  reg [DATA_BITS-1:0] data_in_input;
  wire [DATA_BITS-1:0] data_out_input [TILE_SIZE-1:0];

  reg [TILE_SIZE-1:0] wr_en_result;
  reg [ADDR_BITS-1:0] index_result;
  reg [RES_DATA_BITS-1:0] data_in_result [TILE_SIZE-1:0];
  wire [RES_DATA_BITS-1:0] data_out_result [TILE_SIZE-1:0];

  reg signed [31:0] filter_val_reg;
  reg [15:0] output_ch_idx_reg;
  reg [15:0] input_pixel_idx_reg;

  reg signed [31:0] input_offset_reg;
  reg [9:0] func_id_reg;

  reg signed [31:0] product_reg [TILE_SIZE-1:0];
  reg signed [31:0] offset_product_reg;

  reg signed [31:0] lrelu_pos_mult, lrelu_neg_mult;
  reg signed [5:0]  lrelu_pos_shift, lrelu_neg_shift;
  reg signed [31:0] lrelu_in_offset, lrelu_out_offset;
  reg signed [31:0] lrelu_out_min, lrelu_out_max;

  reg signed [31:0] lrelu_input_reg;
  wire signed [31:0] lrelu_result;
  reg lrelu_start;

  genvar i;

  generate
    for (i = 0; i < TILE_SIZE; i = i + 1)
    begin : gbuff_instances
      global_buffer_bram #(.ADDR_BITS(ADDR_BITS),.DATA_BITS(DATA_BITS)) gbuff_input(
                           .clk(clk), .rst_n(rst_n), .ram_en(1'b1), .wr_en(wr_en_input[i]),
                           .index(index_input), .data_in(data_in_input), .data_out(data_out_input[i]));
    end
  endgenerate

  generate
    for (i = 0; i < TILE_SIZE; i = i + 1)
    begin : result_ram_instances
      result_buffer_lut #(.ADDR_BITS(RESULT_ADDR_BITS), .DATA_BITS(RES_DATA_BITS)) gbuff_result(
                          .clk(clk), .rst_n(rst_n), .ram_en(1'b1), .wr_en(wr_en_result[i]),
                          .index(index_result[RESULT_ADDR_BITS-1:0]), .data_in(data_in_result[i]), .data_out(data_out_result[i]));
    end
  endgenerate

  leaky_relu lrelu_inst (
               .clk(clk), .reset(reset), .start(lrelu_start),
               .x(lrelu_input_reg),
               .pos_multiplier(lrelu_pos_mult), .pos_shift(lrelu_pos_shift),
               .neg_multiplier(lrelu_neg_mult), .neg_shift(lrelu_neg_shift),
               .input_offset(lrelu_in_offset), .output_offset(lrelu_out_offset),
               .output_min(lrelu_out_min), .output_max(lrelu_out_max),
               .result(lrelu_result)
             );

  always @(*)
  begin
    cmd_ready = (State == IDLE);
  end

  always @(posedge clk)
  begin
    if (reset)
      State <= IDLE;
    else
      State <= NextState;
  end

  always @(*)
  begin
    NextState = State;
    case (State)
      IDLE:
      begin
        if (cmd_valid)
        begin
          if (cmd_payload_function_id[9:3] == 0)
            NextState = WRITE_INPUT;
          else if (cmd_payload_function_id[9:3] == 6 || (cmd_payload_function_id[9:3] >= 7 && cmd_payload_function_id[9:3] != 10))
            NextState = RESPOND;
          else
            NextState = PROCESS;
        end
      end

      WRITE_INPUT:
        NextState = IDLE;

      PROCESS:
      begin
        if (func_id_reg[9:3] == 3 || func_id_reg[9:3] == 5 || func_id_reg[9:3] == 10)
          NextState = COMPUTE_MUL;
        else if (func_id_reg[9:3] == 1 || func_id_reg[9:3] == 4)
          NextState = COMPUTE_ADD;
        else
          NextState = RESPOND;
      end

      COMPUTE_MUL:
        NextState = COMPUTE_ADD;

      COMPUTE_ADD:
      begin
        if (func_id_reg[9:3] == 3 || func_id_reg[9:3] == 5)
          NextState = WRITE_BACK;
        else
          NextState = RESPOND;
      end

      WRITE_BACK:
        NextState = RESPOND;
      RESPOND:
        if (rsp_ready)
          NextState = IDLE;
    endcase
  end

  integer k;

  always @(posedge clk)
  begin
    if (reset)
    begin
      rsp_payload_outputs_0 <= 0;
      rsp_valid <= 0;
      filter_val_reg <= 0;
      output_ch_idx_reg <= 0;
      input_pixel_idx_reg <= 0;
      func_id_reg <= 0;
      input_offset_reg <= 0;
      wr_en_input <= 0;
      index_input <= 0;
      data_in_input <= 0;
      wr_en_result <= 0;
      index_result <= 0;

      for(k=0; k<TILE_SIZE; k=k+1)
      begin
        data_in_result[k] <= 0;
        product_reg[k] <= 0;
      end
      offset_product_reg <= 0;

      lrelu_pos_mult <= 0;
      lrelu_pos_shift <= 0;
      lrelu_neg_mult <= 0;
      lrelu_neg_shift <= 0;
      lrelu_in_offset <= 0;
      lrelu_out_offset <= 0;
      lrelu_out_min <= 0;
      lrelu_out_max <= 0;
      lrelu_start <= 0;
      lrelu_input_reg <= 0;
    end
    else
    begin
      wr_en_input <= 0;
      wr_en_result <= 0;
      rsp_valid <= 0;
      lrelu_start <= 0;

      case (State)
        IDLE:
        begin
          rsp_payload_outputs_0 <= 0;

          if (cmd_valid)
          begin
            func_id_reg <= cmd_payload_function_id;

            if (cmd_payload_function_id[9:3] == 0)
            begin
              wr_en_input <= 32'b1 << cmd_payload_inputs_1[31:16];
              index_input <= cmd_payload_inputs_1[15:0];
              data_in_input <= cmd_payload_inputs_0[7:0];
            end
            else if (cmd_payload_function_id[9:3] == 1)
            begin
              index_input <= cmd_payload_inputs_1[15:0];
              input_pixel_idx_reg <= cmd_payload_inputs_0[15:0];
            end
            else if (cmd_payload_function_id[9:3] == 3 || cmd_payload_function_id[9:3] == 5)
            begin
              index_input <= cmd_payload_inputs_1[15:0];
              index_result <= cmd_payload_inputs_1[31:16];
              filter_val_reg <= $signed(cmd_payload_inputs_0);
              output_ch_idx_reg <= cmd_payload_inputs_1[31:16];
            end
            else if (cmd_payload_function_id[9:3] == 4)
            begin
              index_result <= cmd_payload_inputs_0[15:0];
              input_pixel_idx_reg <= cmd_payload_inputs_1[15:0];
            end
            else if (cmd_payload_function_id[9:3] == 6)
            begin
              input_offset_reg <= $signed(cmd_payload_inputs_0);
            end
            else if (cmd_payload_function_id[9:3] == 7)
            begin
              lrelu_pos_mult <= cmd_payload_inputs_0;
              lrelu_pos_shift <= cmd_payload_inputs_1[5:0];
            end
            else if (cmd_payload_function_id[9:3] == 8)
            begin
              lrelu_neg_mult <= cmd_payload_inputs_0;
              lrelu_neg_shift <= cmd_payload_inputs_1[5:0];
            end
            else if (cmd_payload_function_id[9:3] == 9)
            begin
              lrelu_in_offset <= cmd_payload_inputs_0;
              lrelu_out_offset <= cmd_payload_inputs_1;
            end
            else if (cmd_payload_function_id[9:3] == 11)
            begin
              lrelu_out_min <= cmd_payload_inputs_0;
              lrelu_out_max <= cmd_payload_inputs_1;
            end
            else if (cmd_payload_function_id[9:3] == 10)
            begin
              lrelu_input_reg <= $signed(cmd_payload_inputs_0);
            end
          end
        end

        WRITE_INPUT:
        begin
          rsp_valid <= 1;
        end

        PROCESS:
        begin
        end

        COMPUTE_MUL:
        begin
          for (k = 0; k < TILE_SIZE; k = k + 1)
          begin
            product_reg[k] <= $signed(data_out_input[k]) * filter_val_reg;
          end
          offset_product_reg <= input_offset_reg * filter_val_reg;

          if (func_id_reg[9:3] == 10)
          begin
            lrelu_start <= 1;
          end
        end

        COMPUTE_ADD:
        begin
          if (func_id_reg[9:3] == 3)
          begin
            for (k = 0; k < TILE_SIZE; k = k + 1)
            begin
              data_in_result[k] <= $signed(data_out_result[k]) + product_reg[k] + offset_product_reg;
            end
            index_result <= output_ch_idx_reg;
            wr_en_result <= {TILE_SIZE{1'b1}};
          end
          else if (func_id_reg[9:3] == 5)
          begin
            for (k = 0; k < TILE_SIZE; k = k + 1)
            begin
              data_in_result[k] <= product_reg[k] + offset_product_reg;
            end
            index_result <= output_ch_idx_reg;
            wr_en_result <= {TILE_SIZE{1'b1}};
          end
        end

        WRITE_BACK:
        begin
        end

        RESPOND:
        begin
          rsp_valid <= 1;
          if (func_id_reg[9:3] == 1)
            rsp_payload_outputs_0 <= {24'b0, data_out_input[input_pixel_idx_reg[RESULT_ADDR_BITS-1:0]]};
          else if (func_id_reg[9:3] == 4)
            rsp_payload_outputs_0 <= data_out_result[input_pixel_idx_reg[RESULT_ADDR_BITS-1:0]];
          else if (func_id_reg[9:3] == 10)
            rsp_payload_outputs_0 <= lrelu_result;
          else
            rsp_payload_outputs_0 <= 0;
        end
      endcase
    end
  end
endmodule
