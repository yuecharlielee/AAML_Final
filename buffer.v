module buffer (
    input clk,
    input rst_n,
    input in_valid,
    input [31:0] data_in,
    output reg [31:0] data_out
);

reg [7:0] r1_temp1;
reg [7:0] r2_temp1, r2_temp2;
reg [7:0] r3_temp1, r3_temp2, r3_temp3;


always @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
        data_out <= 32'b0;
        r1_temp1 <= 8'b0;
        r2_temp1 <= 8'b0;
        r2_temp2 <= 8'b0;
        r3_temp1 <= 8'b0;
        r3_temp2 <= 8'b0;
        r3_temp3 <= 8'b0;
    end 
    else if(in_valid) begin
        data_out <= 32'b0;
        r1_temp1 <= 8'b0;
        r2_temp1 <= 8'b0;
        r2_temp2 <= 8'b0;
        r3_temp1 <= 8'b0;
        r3_temp2 <= 8'b0;
        r3_temp3 <= 8'b0;
    end
    else begin
        r1_temp1 <= data_in[23:16];

        r2_temp1 <= r2_temp2;
        r2_temp2 <= data_in[15:8];

        r3_temp1 <= r3_temp2;
        r3_temp2 <= r3_temp3;
        r3_temp3 <= data_in[7:0];

        data_out <= {data_in[31:24], r1_temp1, r2_temp1, r3_temp1};

    end
end 

endmodule