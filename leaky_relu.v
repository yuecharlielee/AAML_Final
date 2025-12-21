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

    reg signed [63:0] product;
    reg signed [31:0] scaled_result;
    reg signed [31:0] unclamped_output;

    always @(*) begin
        product = $signed(x_adjusted) * $signed(selected_multiplier);

        if ($signed(selected_shift) >= 0) begin
            scaled_result = (product >>> 31) <<< selected_shift;
        end else begin
            scaled_result = (product >>> 31) >>> (-$signed(selected_shift));
        end

        unclamped_output = scaled_result + output_offset;

        if (unclamped_output > output_max) begin
            result = output_max;
        end else if (unclamped_output < output_min) begin
            result = output_min;
        end else begin
            result = unclamped_output;
        end
    end

endmodule