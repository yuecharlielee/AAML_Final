module PE(
    rst_n,
    in_valid,
    clk,
    up_in,
    left_in,
    right_out,
    down_out,
    result_out,
    inputoffset
);

input clk;
input rst_n;
input [7:0] up_in, left_in;
input in_valid;
input [31:0] inputoffset;

wire signed [8:0] left_in_ext = $signed(left_in);
wire signed [8:0] left_in_offset = left_in_ext + $signed(inputoffset[8:0]); 
wire signed [8:0] up_in_ext = $signed(up_in);
wire signed [8:0] up_in_offset = up_in_ext + $signed(inputoffset[8:0]);


output reg [7:0] right_out, down_out;
output reg signed [31:0] result_out;


always @(posedge clk or negedge rst_n) begin
    if(!rst_n) begin
        result_out <= 0;
        right_out <= 0;
        down_out <= 0;
    end
    else if(in_valid) begin
        result_out <= 0;
        right_out <= 0;
        down_out <= 0;
    end
    else begin
        result_out <= $signed(result_out) + ($signed(up_in_offset)) * ($signed(left_in_ext));
        right_out <= left_in;
        down_out <= up_in;
    end
end



endmodule