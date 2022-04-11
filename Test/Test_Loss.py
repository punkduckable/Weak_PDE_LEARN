# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# external libraries and stuff.
import numpy;
import torch;
import unittest;
import random;
import math;

# Code files.
from Loss import Coll_Loss, L0_Approx_Loss, Lp_Loss;
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


    def test_L0_Approx_Loss(self):
        ########################################################################
        # Test 1 : Xi = 0

        # Instantiate Xi.
        N : int = random.randrange(5, 100);
        Xi      = torch.empty(N, dtype = torch.float32);
        s       = random.uniform(.01, .1);

        # First, try with the zero vector.
        for i in range(N):
            Xi[i] = 0;

        # In this case, we expect the approximation to the L0 norm to give zero.
        Predict : float = 0;
        Actual  : float = L0_Approx_Loss(Xi = Xi, s = s).item();

        epsilon : float = .00001;
        self.assertLess(abs(Predict - Actual), epsilon);



        ########################################################################
        # Test 2 : All componnts of Xi are the same.

        # Now replace Xi with a randomly selected value.
        x  = random.uniform(.5, 1.5);
        Xi = torch.full_like(Xi, x);

        # In this case, we expect the result to be N(1 - exp(-x^2/s^2)).
        Predict = N*(1 - math.exp(-(x**2)/(s**2)));
        Actual  = L0_Approx_Loss(Xi = Xi, s = s).item();

        self.assertLess(abs(Predict - Actual), epsilon);



        ########################################################################
        # Test 3: Fill some, but not all, of Xi's components.

        Half_N = N // 2;
        Xi = torch.zeros_like(Xi);
        y = random.uniform(.01, .1);
        for i in range(Half_N):
            Xi[i] = y;

        # We now expect the result to be Half_N*(1 - exp(-y^2/(s**2))) (think about it).
        Predict = Half_N*(1 - math.exp(-(y**2)/(s**2)));
        Actual  = L0_Approx_Loss(Xi = Xi, s = s).item();

        self.assertLess(abs(Predict - Actual), epsilon);
        #print("s = %f, y = %f, N = %d, Predict = %lf, actual = %f" % (s, y, N, Predict, Actual));
