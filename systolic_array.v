module systolic_array(
    rst_n,
    in_valid,
    clk,
    in_a,
    in_b,
    out_c_r1,
    out_c_r2,
    out_c_r3,
    out_c_r4,
    inputoffset
);

input rst_n;
input in_valid;
input clk;
input [31:0] in_a;
input [31:0] in_b;
input [31:0] inputoffset;

output [127:0] out_c_r1; 
output [127:0] out_c_r2;
output [127:0] out_c_r3;
output [127:0] out_c_r4;


wire [7:0] vertical_wire[4:0][3:0]; 
wire [7:0] horizontal_wire[3:0][4:0];  
wire [31:0] c_wire[3:0][3:0]; 


assign vertical_wire[0][0] = in_b[31:24];  
assign vertical_wire[0][1] = in_b[23:16];  
assign vertical_wire[0][2] = in_b[15:8];  
assign vertical_wire[0][3] = in_b[7:0];   


assign horizontal_wire[0][0] = in_a[31:24];  
assign horizontal_wire[1][0] = in_a[23:16];  
assign horizontal_wire[2][0] = in_a[15:8];   
assign horizontal_wire[3][0] = in_a[7:0];    

genvar i,j;



generate
    for(i=0;i<4;i=i+1) begin:ROW
        for(j=0;j<4;j=j+1) begin:COL
            PE pe(
                .rst_n(rst_n),
                .in_valid(in_valid),
                .clk(clk),
                .up_in(vertical_wire[i][j]),
                .left_in(horizontal_wire[i][j]),
                .right_out(horizontal_wire[i][j+1]),
                .down_out(vertical_wire[i+1][j]),
                .result_out(c_wire[i][j]),
                .inputoffset(inputoffset)
            );
        end
    end
endgenerate

assign out_c_r1 = {c_wire[0][0], c_wire[0][1], c_wire[0][2], c_wire[0][3]};
assign out_c_r2 = {c_wire[1][0], c_wire[1][1], c_wire[1][2], c_wire[1][3]};
assign out_c_r3 = {c_wire[2][0], c_wire[2][1], c_wire[2][2], c_wire[2][3]};
assign out_c_r4 = {c_wire[3][0], c_wire[3][1], c_wire[3][2], c_wire[3][3]};

endmodule