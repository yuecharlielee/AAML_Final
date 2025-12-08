module leaky_relu(
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


    wire signed [31:0] x_adjusted;
    assign x_adjusted = x - input_offset;

    wire signed [31:0] selected_multiplier;
    wire signed [ 5:0] selected_shift;
    
    assign selected_multiplier = (x_adjusted[31]) ? neg_multiplier : pos_multiplier;
    assign selected_shift      = (x_adjusted[31]) ? neg_shift      : pos_shift;


    reg signed [31:0] s1_x_shifted;
    reg signed [ 5:0] s1_right_shift;
    
    reg signed [63:0] ab_64;
    reg signed [63:0] nudge;
    reg signed [63:0] acc_64;
    reg signed [31:0] srdhm_result; 
    
    reg [31:0] mask;
    reg [31:0] remainder;
    reg [31:0] threshold;
    reg        adjustment;
    reg signed [31:0] raw_shifted;
    
    reg signed [31:0] unclamped_output;

    always @(*) begin

        if ($signed(selected_shift) > 0) begin
            s1_x_shifted   = x_adjusted <<< selected_shift;
            s1_right_shift = 6'd0;
        end else begin
            s1_x_shifted   = x_adjusted;
            s1_right_shift = -$signed(selected_shift);
        end
        

        ab_64 = $signed(s1_x_shifted) * $signed(selected_multiplier);


        if (ab_64 >= 0) 
            nudge = 64'h00000000_40000000;
        else 
            nudge = 64'hFFFFFFFF_C0000001; 

        acc_64 = ab_64 + nudge;

        if ((s1_x_shifted == 32'h80000000) && (selected_multiplier == 32'h80000000)) begin
            srdhm_result = 32'h7FFFFFFF; 
        end else begin
            if (acc_64 < 0 && acc_64[30:0] != 0) begin
                srdhm_result = acc_64[62:31] + 32'd1;
            end else begin
                srdhm_result = acc_64[62:31];
            end
        end

        mask = (32'd1 << s1_right_shift) - 32'd1;
        

        remainder = srdhm_result & mask;

        threshold = (mask >> 1) + ((srdhm_result < 0) ? 32'd1 : 32'd0);
        
        adjustment = (remainder > threshold);
        
        raw_shifted = srdhm_result >>> s1_right_shift;

        unclamped_output = raw_shifted + {31'd0, adjustment} + output_offset;

        if (unclamped_output > output_max) begin
            result = output_max;
        end else if (unclamped_output < output_min) begin
            result = output_min;
        end else begin
            result = unclamped_output;
        end
    end

endmodule