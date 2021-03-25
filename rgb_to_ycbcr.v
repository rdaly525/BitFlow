`include "svreal.sv"
module rgb_to_ycbcr #(
	`DECL_REAL(r),
	`DECL_REAL(g),
	`DECL_REAL(b), 
	`DECL_REAL(col_1), 
	`DECL_REAL(col_2), 
	`DECL_REAL(col_3)) 
(
	`INPUT_REAL(r), 
	`INPUT_REAL(g), 
	`INPUT_REAL(b), 
	`OUTPUT_REAL(col_1), 
	`OUTPUT_REAL(col_2), 
	`OUTPUT_REAL(col_3)
); 
 
	`MAKE_CONST_REAL(0.299, C1);
	`MAKE_CONST_REAL(0.587, C2);
	`MAKE_CONST_REAL(0.114, C3);
	`MAKE_CONST_REAL(-0.16875, C4);
	`MAKE_CONST_REAL(-0.33126, C5);
	`MAKE_CONST_REAL(0.5, C6);
	`MAKE_CONST_REAL(0.5, C7);
	`MAKE_CONST_REAL(-0.41869, C8);
	`MAKE_CONST_REAL(-0.08131, C9);

	`MAKE_GENERIC_REAL(C9_mul_b, -1024.0, 30);
	`MUL_INTO_REAL(C9, b, C9_mul_b);

	`MAKE_GENERIC_REAL(C8_mul_g, -1024.0, 30);
	`MUL_INTO_REAL(C8, g, C8_mul_g);

	`MAKE_GENERIC_REAL(C7_mul_r, -1024.0, 30);
	`MUL_INTO_REAL(C7, r, C7_mul_r);

	`MAKE_GENERIC_REAL(C7_mul_r_add_C8_mul_g, -1024.0, 30);
	`ADD_INTO_REAL(C7_mul_r, C8_mul_g, C7_mul_r_add_C8_mul_g);

	`ADD_INTO_REAL(C7_mul_r_add_C8_mul_g, C9_mul_b, col_3);

	`MAKE_GENERIC_REAL(C6_mul_b, -1024.0, 30);
	`MUL_INTO_REAL(C6, b, C6_mul_b);

	`MAKE_GENERIC_REAL(C5_mul_g, -1024.0, 30);
	`MUL_INTO_REAL(C5, g, C5_mul_g);

	`MAKE_GENERIC_REAL(C4_mul_r, -1024.0, 30);
	`MUL_INTO_REAL(C4, r, C4_mul_r);

	`MAKE_GENERIC_REAL(C4_mul_r_add_C5_mul_g, -1024.0, 30);
	`ADD_INTO_REAL(C4_mul_r, C5_mul_g, C4_mul_r_add_C5_mul_g);

	`ADD_INTO_REAL(C4_mul_r_add_C5_mul_g, C6_mul_b, col_2);

	`MAKE_GENERIC_REAL(C3_mul_b, -1024.0, 30);
	`MUL_INTO_REAL(C3, b, C3_mul_b);

	`MAKE_GENERIC_REAL(C2_mul_g, -1024.0, 30);
	`MUL_INTO_REAL(C2, g, C2_mul_g);

	`MAKE_GENERIC_REAL(C1_mul_r, -1024.0, 30);
	`MUL_INTO_REAL(C1, r, C1_mul_r);

	`MAKE_GENERIC_REAL(C1_mul_r_add_C2_mul_g, -1024.0, 30);
	`ADD_INTO_REAL(C1_mul_r, C2_mul_g, C1_mul_r_add_C2_mul_g);

	`ADD_INTO_REAL(C1_mul_r_add_C2_mul_g, C3_mul_b, col_1);


endmodule