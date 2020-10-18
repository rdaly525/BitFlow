module rgb_to_ycbcr (input [28:0] C1, input [24:0] r, input [28:0] C2, input [24:0] g, input [28:0] C3, input [24:0] b, input [28:0] C4, input [28:0] C5, input [28:0] C6, input [28:0] C7, input [28:0] C8, input [28:0] C9, output [15.0:0] col_1, output [15.0:0] col_2, output [15.0:0] col_3); 
 
	wire [29:0] C9_mul_b;
	assign C9_mul_b = C9 * b;

	wire [29:0] C8_mul_g;
	assign C8_mul_g = C8 * g;

	wire [29:0] C7_mul_r;
	assign C7_mul_r = C7 * r;

	wire [29:0] C7_mul_r_add_C8_mul_g;
	assign C7_mul_r_add_C8_mul_g = C7_mul_r + C8_mul_g;

	assign col_3 = C7_mul_r_add_C8_mul_g + C9_mul_b;

	wire [29:0] C6_mul_b;
	assign C6_mul_b = C6 * b;

	wire [29:0] C5_mul_g;
	assign C5_mul_g = C5 * g;

	wire [29:0] C4_mul_r;
	assign C4_mul_r = C4 * r;

	wire [29:0] C4_mul_r_add_C5_mul_g;
	assign C4_mul_r_add_C5_mul_g = C4_mul_r + C5_mul_g;

	assign col_2 = C4_mul_r_add_C5_mul_g + C6_mul_b;

	wire [29:0] C3_mul_b;
	assign C3_mul_b = C3 * b;

	wire [29:0] C2_mul_g;
	assign C2_mul_g = C2 * g;

	wire [29:0] C1_mul_r;
	assign C1_mul_r = C1 * r;

	wire [29:0] C1_mul_r_add_C2_mul_g;
	assign C1_mul_r_add_C2_mul_g = C1_mul_r + C2_mul_g;

	assign col_1 = C1_mul_r_add_C2_mul_g + C3_mul_b;


endmodule