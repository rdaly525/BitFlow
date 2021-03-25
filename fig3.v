`include "svreal.sv"
module fig3 #(
`DECL_REAL(a),
`DECL_REAL(b), 
`DECL_REAL(z)) 
(
`INPUT_REAL(a), 
`INPUT_REAL(b), 
`OUTPUT_REAL(z)
); 
 
`MAKE_CONST_REAL(4.3, c);

`MAKE_GENERIC_REAL(d, -16.0, 17);
`MUL_INTO_REAL(a, b, d);

`MAKE_GENERIC_REAL(e, -16.0, 18);
`ADD_INTO_REAL(d, c, e);

`SUB_INTO_REAL(e, b, z);


endmodule