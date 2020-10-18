module fig3 (input [17:0] a, input [17:0] b, input [19:0] c, output [13.0:0] z); 
 
	wire [19:0] d;
	assign d = a * b;

	wire [18:0] e;
	assign e = d + c;

	assign z = e - b;


endmodule