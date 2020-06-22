from Curis.caseStudies import caseStudy

def test_poly_approx():

    tmp=caseStudy
    print((tmp.poly_approx([1,2,3],3,5)))
    print((tmp.poly_approx([1,2,3,4],4,5)))
    print((tmp.poly_approx([1,2],2,4)))

def test_RGB_to_YCbCr():
    
    tmp=caseStudy
    print((tmp.RGB_to_YCbCr([[1],[1],[1]])))
    print((tmp.RGB_to_YCbCr([[122],[23],[56]])))

def test_Matrix_Multiplication():

    tmp=caseStudy
    print((tmp.Matrix_Multiplication([[1,2],[3,4]],[[5,6],[7,8]])))
    print((tmp.Matrix_Multiplication([[1,2],[6,5]],[[9,3],[1,12]])))
   

    
    
    


