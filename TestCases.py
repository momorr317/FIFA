#Runhao Zhao (rz6dg) Wenxi Zhao (wz8nx) Shaoran Li (sl4bz) Winfred Hill (whh3rz) Jingnan Yang (jy4fch)
from project import *
import unittest

fifa = fifaDataAnalysis()

# Class for testing the wage method
class test_wage(unittest.TestCase):
    # Test when the wage is in normal range
    def test_response1(self):
        response = fifa.Wage(350)
        self.assertEqual(response, "300K-400K")
    # Test when the wage is on the margin
    def test_response2(self):
        response = fifa.Wage(500)
        self.assertEqual(response, "400K-500K")
    # Test when the wage is at minimum
    def test_response3(self):
        response = fifa.Wage(0)
        self.assertEqual(response, "<100K")
    # Test when the wage is above the maximum  
    def test_response4(self):
        response = fifa.Wage(750)
        self.assertEqual(response, "")
        
if __name__ == '__main__':
    unittest.main()
    
# Class for testing the position method
class test_position(unittest.TestCase):
    # Test when the input is "GK"
    def test_response1(self):
        response = fifa.position("GK")
        self.assertEqual(response, "GK")
    # Test when the input is in DEF
    def test_response2(self):
        response = fifa.position("LB")
        self.assertEqual(response, "DEF")
    # Test when the input is in LM
    def test_response3(self):
        response = fifa.position("LM")
        self.assertEqual(response, "MID")
    # Test when the input is in FWD
    def test_response4(self):
        response = fifa.position("CF")
        self.assertEqual(response, "FWD")
       
# Test program
def test():
    # Test the wage method
    wageTest = test_wage()
    if wageTest.test_response1() == wageTest.test_response2() \
       == wageTest.test_response3() == wageTest.test_response4() == None:
        print("The wage method passed the test")
    else:
        print("The wage method did not pass the test")
    
    # Test the position method
    positionTest = test_position()
    if positionTest.test_response1() == positionTest.test_response2() \
       == positionTest.test_response3() == positionTest.test_response4() == None:
        print("The position method passed the test")
    else:
        print("The position method did not pass the test")
 
# Run the tests       
test()
if __name__ == '__main__':
    unittest.main()