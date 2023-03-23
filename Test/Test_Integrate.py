# Nonsense to add Code directory to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Get the path to the Code, Classes directories.
Code_Path       = os.path.join(Main_Path, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add these directories to Python's search path.
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# external libraries and stuff.
import numpy;
import torch;
import unittest;
import random;
import math;
import time;

# Code files.
from Integrate          import Integrate_PDE;
from Weight_Function    import Weight_Function;
from Derivative         import Derivative;
from Trial_Function     import Trial_Function;
from Library_Term       import Library_Term;




# We need to define a Module object that can take the place of U. 
class Polynomial(torch.nn.Module):
    """
    This class defines a Module that acts something like a Polynomial. Each 
    polynomial object has two attributes, Powers and Offsets. Both are a 1D 
    numpy ndarrays. 
    
    If Powers/Offsets have n entries, then the corresponding polynomial object 
    is a function of n variables. For brevity, let P = Powers and O = Offsets. 
    Then, the polynomial object defines the following function:
        f(x_0, ... , x_{n - 1}) = \prod_{k = 0}^{n - 1} (x_k - O[k])^P[k]
    """

    def __init__(self, Powers : numpy.ndarray, Offsets : numpy.ndarray):
        """
        This function sets up a Polynomial object!
        """

        # Make sure the arguments are valid. 
        assert(len(Powers.shape) == 1);
        assert(Powers.shape == Offsets.shape);

        # Call the Module class initializer.
        super(Polynomial, self).__init__();

        # Set up self's attributes. 
        self.n         = Powers.size;
        self.Powers     = Powers;
        self.Offsets    = Offsets;

         
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """
        This function evaluates self at each row of X. We assume that X is a
        2D Tensor with self.n columns. 
        """

        assert(X.shape[1] == self.n);

        Prod : torch.Tensor = torch.ones(X.shape[0], dtype = torch.float32);

        # Compute the polynomial component-by-component
        for k in range(self.n):
            X_k             : torch.Tensor = X[:, k];
            X_k_minus_O_k   : torch.Tensor = X_k - self.Offsets[k];

            Prod *= torch.pow(X_k_minus_O_k, self.Powers[k]);
        
        return Prod;



def Test_Integrate() -> None:
    """ This function tests the Integrate function. To do this, we integrate
    various derivatives of a weight function. Currently, it is up to the user
    to decide if those values are correct. """

    print("Setting up...  ", end = '');
    timer       : float = time.perf_counter();

    # First, set up a grid.
    n       : int               = 3;
    Bounds  : numpy.ndarray     = numpy.empty(shape = [n, 2], dtype = numpy.float32);
    Bounds[:, 0]                = -3;
    Bounds[:, 1]                = 3;

    Gridlines_Per_Axis : int    = 200;
    Coords  : torch.Tensor      = torch.empty([Gridlines_Per_Axis, Gridlines_Per_Axis, Gridlines_Per_Axis, n], dtype = torch.float32);

    x_Values : numpy.ndarray    = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Gridlines_Per_Axis, dtype = numpy.float32);
    y_Values : numpy.ndarray    = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Gridlines_Per_Axis, dtype = numpy.float32);
    z_Values : numpy.ndarray    = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Gridlines_Per_Axis, dtype = numpy.float32);

    for t in range(Gridlines_Per_Axis):
        Coords[t, :, :, 0] = x_Values[t].item();
        Coords[:, t, :, 1] = y_Values[t].item();
        Coords[:, :, t, 2] = z_Values[t].item();
    
    # Reshape into a 2D array, with one coordinate per row.
    Coords = Coords.view(-1, n);

    # Determine sub-rectangle volume.
    x_spacing   : float         = x_Values[1] - x_Values[0];
    y_spacing   : float         = y_Values[1] - y_Values[0];
    z_spacing   : float         = z_Values[1] - z_Values[0];
    V           : float         = x_spacing*y_spacing*z_spacing;

    # Next, set up the weight function.
    r   : int               = 1;
    X_0 : numpy.ndarray     = torch.tensor([0, 0, 0], dtype = torch.float32);
    w   : Weight_Function   = Weight_Function(X_0 = X_0, r = r, Coords = Coords, V = V);

    # Set up a Derivative operator.
    Encoding : numpy.ndarray = numpy.array([0, 1, 0], dtype = numpy.int32);
    D : Derivative = Derivative(Encoding = Encoding);    

    # Calculate D(w).
    w.Add_Derivative(D = D);

    # Compute the polynomial of X. 
    P = Polynomial(Powers = numpy.array([2, 1, 2]), Offsets = X_0);

    # Set up the library terms.
    LHS_Term    = Library_Term(Derivative = D, Trial_Function = Trial_Function(Power = 1));
    RHS_Terms   = [LHS_Term];

    # Finally, set up the mask
    Mask = torch.zeros(len(RHS_Terms), dtype = torch.bool);

    # All done with setup!
    print("Done! Took %fs" % (time.perf_counter() - timer));



    ###########################################################################
    # Integrate!
    print("Integrating... ", end = '');
    timer       : float = time.perf_counter();
    Integral    : float = Integrate_PDE(w = w, U = P, LHS_Term = LHS_Term, RHS_Terms = RHS_Terms, Mask = Mask)[0].item();
    print("Done! Took %fs" % (time.perf_counter() - timer));
    print(Integral);



if __name__ == "__main__":
    Test_Integrate();
