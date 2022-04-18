# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_Path   = os.path.join(Main_Path, "Code");
sys.path.append(Code_Path);

# external libraries and stuff.
import numpy;
import torch;
import unittest;
import random;
import math;

# Code files.
from Loss   import Lp_Loss;
from Points import Generate_Points;

# Other test file.
from Polynomials import Polynomial_1d, Polynomial_2d;


class Loss_Test(unittest.TestCase):
    def test_Lp_Loss(self):
        ########################################################################
        # Test 1: Xi = 0.

        # Instantiate Xi.
        N : int   = random.randrange(5, 100);
        Xi        = torch.zeros(N, dtype = torch.float32);
        p : float = random.uniform(.01, .1);

        # In this case, we expect the Lp loss to be 0.
        Predict : float = 0;
        Actual  : float = Lp_Loss(Xi = Xi, p = p).item();

        # Check results
        epsilon : float = .00001;
        self.assertLess(abs(Predict - Actual), epsilon);


        ########################################################################
        # Test 2 : All components of Xi are the same.

        # Now replace Xi with a randomly selected value.
        x  = random.uniform(.01, .1);
        Xi = torch.full_like(Xi, x);

        # In this case, we expect the result to be N*(x^p).
        Predict = N*(x ** p);
        Actual  = Lp_Loss(Xi = Xi, p = p).item();

        # Check results
        self.assertLess(abs(Predict - Actual), epsilon);

        ########################################################################
        # Test 3 : a random number of components of Xi are the same, the rest
        # are zero.

        M : int = random.randrange(1, N - 1);
        Xi[M:] = 0;

        # In this case, we expect the result to be M*(x^p).
        Predict = M*(x **p );
        Actual = Lp_Loss(Xi = Xi, p = p).item();

        self.assertLess(abs(Predict - Actual), epsilon);
        #print("p = %f, x = %f, M = %d, Predict = %lf, actual = %f" % (p, x, M, Predict, Actual));
