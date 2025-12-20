

module global_buffer_bram #(parameter ADDR_BITS=8, parameter DATA_BITS=8)(
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

  always @ (posedge clk) begin
    if (ram_en) begin
      if(wr_en) begin
        gbuff[index] <= data_in;
      end
      else begin
        data_out <= gbuff[index];
      end
    end
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
  
  localparam RESULT_ADDR_BITS = 5;
  localparam ADDR_BITS = 14;
  localparam DATA_BITS = 8;
  localparam RES_DATA_BITS = 32;
  localparam Tile_size = 32;

  reg [5:0] State, NextState;
  localparam IDLE         = 6'd0,
             WRITE_INPUT  = 6'd1, 
             PROCESS      = 6'd2, 
             COMPUTE_MUL  = 6'd3, 
             COMPUTE_ADD  = 6'd4, 
             WRITE_BACK   = 6'd5, 
             RESPOND      = 6'd6;

  reg [Tile_size-1:0] wr_en_input;
  reg [ADDR_BITS-1:0] index_input;
  reg [DATA_BITS-1:0] data_in_input;
  wire [DATA_BITS-1:0] data_out_input [31:0];

  reg [Tile_size-1:0] wr_en_result;
  reg [ADDR_BITS-1:0] index_result;
  reg [RES_DATA_BITS-1:0] data_in_result [31:0];
  wire [RES_DATA_BITS-1:0] data_out_result [31:0];

  reg signed [31:0] filter_val_reg;
  reg [15:0] output_ch_idx_reg;
  reg [15:0] input_pixel_idx_reg; 
  
  // [新增] Input Offset 暫存器
  reg signed [31:0] input_offset_reg;
  
  reg [9:0] func_id_reg;
  reg signed [31:0] product_reg [31:0];

  genvar i;
  
  // Input Buffer (Block RAM)
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

  // Result Buffer (Distributed RAM)
  generate
    for (i = 0; i < 32; i = i + 1) begin : result_ram_instances
        global_buffer_bram #(
            .ADDR_BITS(RESULT_ADDR_BITS), 
            .DATA_BITS(RES_DATA_BITS) 
        ) gbuff_result(
            .clk(clk),
            .rst_n(rst_n),
            .ram_en(1'b1),
            .wr_en(wr_en_result[i]),
            .index(index_result[RESULT_ADDR_BITS-1:0]), 
            .data_in(data_in_result[i]),
            .data_out(data_out_result[i])
        );
    end
  endgenerate

  always @(*) begin
    cmd_ready = (State == IDLE);
  end

  always @(posedge clk) begin
    if (reset)
      State <= IDLE;
    else
      State <= NextState;
  end

  // State Machine
  always @(*) begin
    NextState = State;
    case (State)
      IDLE: begin
        if (cmd_valid) begin
            if (cmd_payload_function_id[9:3] == 0)
                NextState = WRITE_INPUT;
            else if (cmd_payload_function_id[9:3] == 6) // Op 6: Write Offset -> RESPOND
                NextState = RESPOND;
            else
                NextState = PROCESS;
        end
      end
      
      WRITE_INPUT: begin
          NextState = IDLE; // Fast Ack
      end

      PROCESS: begin
        // Op 3 (Accum), Op 5 (Overwrite)
        if (func_id_reg[9:3] == 3 || func_id_reg[9:3] == 5)
            NextState = COMPUTE_MUL;
        // Op 1 (Read Input), Op 4 (Read Result)
        else if (func_id_reg[9:3] == 1 || func_id_reg[9:3] == 4)
            NextState = COMPUTE_ADD; 
        else
            NextState = RESPOND;
      end

      COMPUTE_MUL: begin
        NextState = COMPUTE_ADD;
      end

      COMPUTE_ADD: begin
        if (func_id_reg[9:3] == 3 || func_id_reg[9:3] == 5)
             NextState = WRITE_BACK;
        else 
             NextState = RESPOND;
      end

      WRITE_BACK: begin
        NextState = RESPOND;
      end

      RESPOND: begin
        if (rsp_ready)
          NextState = IDLE;
      end
    endcase
  end

  integer k;
  
  always @(posedge clk) begin
    if (reset) begin
        rsp_payload_outputs_0 <= 0;
        rsp_valid <= 0;
        filter_val_reg <= 0;
        output_ch_idx_reg <= 0;
        input_pixel_idx_reg <= 0;
        func_id_reg <= 0;
        input_offset_reg <= 0; // Reset Offset
        
        wr_en_input <= 0;
        index_input <= 0;
        data_in_input <= 0;
        
        wr_en_result <= 0;
        index_result <= 0;
        for(k=0; k<32; k=k+1) begin
            data_in_result[k] <= 0;
            product_reg[k] <= 0;
        end
    end
    else begin
        wr_en_input <= 0;
        wr_en_result <= 0; 
        rsp_valid <= 0;

        case (State)
            IDLE: begin
                rsp_payload_outputs_0 <= 0;
                
                if (cmd_valid) begin
                    func_id_reg <= cmd_payload_function_id;

                    // === Op 0: Write Input ===
                    if (cmd_payload_function_id[9:3] == 0) begin 
                        wr_en_input <= 32'b1 << cmd_payload_inputs_1[31:16];
                        index_input <= cmd_payload_inputs_1[15:0];
                        data_in_input <= cmd_payload_inputs_0[7:0];
                    end
                    
                    // === Op 1: Read Input ===
                    else if (cmd_payload_function_id[9:3] == 1) begin
                        index_input <= cmd_payload_inputs_1[15:0];
                        input_pixel_idx_reg <= cmd_payload_inputs_0[15:0]; 
                    end
                    
                    // === Op 3 (Accum) / Op 5 (Overwrite) ===
                    else if (cmd_payload_function_id[9:3] == 3 || cmd_payload_function_id[9:3] == 5) begin
                        index_input <= cmd_payload_inputs_1[15:0];
                        index_result <= cmd_payload_inputs_1[31:16]; 
                        filter_val_reg <= $signed(cmd_payload_inputs_0);
                        output_ch_idx_reg <= cmd_payload_inputs_1[31:16]; 
                    end
                    
                    // === Op 4: Read Result ===
                    else if (cmd_payload_function_id[9:3] == 4) begin
                        index_result <= cmd_payload_inputs_0[15:0]; 
                        input_pixel_idx_reg <= cmd_payload_inputs_1[15:0]; 
                    end

                    // === [新增] Op 6: Set Input Offset ===
                    else if (cmd_payload_function_id[9:3] == 6) begin
                        input_offset_reg <= $signed(cmd_payload_inputs_0);
                    end
                end
            end
            
            WRITE_INPUT: begin
                rsp_valid <= 1; // Ack for Op 0
            end

            PROCESS: begin
                // BRAM Read
            end

            COMPUTE_MUL: begin
                // Pipeline Stage 1: Multiply with Offset
                // Formula: (Input + Offset) * Weight
                for (k = 0; k < 32; k = k + 1) begin
                    product_reg[k] <= ($signed(data_out_input[k]) + input_offset_reg) * filter_val_reg;
                end
            end

            COMPUTE_ADD: begin
                // Pipeline Stage 2: Add
                if (func_id_reg[9:3] == 3) begin
                    for (k = 0; k < 32; k = k + 1) begin
                        data_in_result[k] <= $signed(data_out_result[k]) + product_reg[k];
                    end
                    index_result <= output_ch_idx_reg;
                    wr_en_result <= {32{1'b1}}; 
                end
                else if (func_id_reg[9:3] == 5) begin
                    for (k = 0; k < 32; k = k + 1) begin
                        data_in_result[k] <= product_reg[k]; 
                    end
                    index_result <= output_ch_idx_reg;
                    wr_en_result <= {32{1'b1}}; 
                end
            end
            
            WRITE_BACK: begin
            end

            RESPOND: begin
                rsp_valid <= 1;

                if (func_id_reg[9:3] == 1) begin
                     rsp_payload_outputs_0 <= {24'b0, data_out_input[input_pixel_idx_reg[4:0]]};
                end
                else if (func_id_reg[9:3] == 4) begin
                    rsp_payload_outputs_0 <= data_out_result[input_pixel_idx_reg[4:0]];
                end
                // Op 6 (Set Offset) also returns 0 here
                else begin
                    rsp_payload_outputs_0 <= 0;
                end
            end
        endcase
    end
  end

endmodule