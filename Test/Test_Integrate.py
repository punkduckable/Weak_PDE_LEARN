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
import time;

# Code files.
from Integrate          import Integrate;
from Weight_Function    import Weight_Function;
from Derivative         import Derivative;
from Trial_Function     import Trial_Function;



def Test_Integrate() -> None:
    """ This function tests the Integrate function. To do this, we integrate
    various derivatives of a weight function. Currently, it is up to the user
    to decide if those values are correct. """

    print("Setting up... ", end = '');

    # First, set up a grid.
    n       : int               = 3;
    Bounds  : numpy.ndarray     = numpy.empty(shape = [n, 2], dtype = numpy.float32);
    Bounds[:, 0]                = -3;
    Bounds[:, 1]                = 3;

    Gridlines_per_axis : int    = 200;
    Coords  : torch.Tensor      = torch.empty([Gridlines_per_axis, Gridlines_per_axis, Gridlines_per_axis, n], dtype = torch.float32);

    x_Values : numpy.ndarray    = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Gridlines_per_axis, dtype = numpy.float32);
    y_Values : numpy.ndarray    = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Gridlines_per_axis, dtype = numpy.float32);
    z_Values : numpy.ndarray    = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Gridlines_per_axis, dtype = numpy.float32);

    for i in range(Gridlines_per_axis):
        Coords[i, :, :, 0] = x_Values[i].item();

        for j in range(Gridlines_per_axis):
            Coords[i, j, :, 1] = y_Values[j].item();

            for k in range(Gridlines_per_axis):
                Coords[i, j, k, 2] = z_Values[k].item();

    # Reshape into a 2D array
    Coords = Coords.view(-1, n);

    # Determine sub-rectangle volume.
    x_spacing   : float         = x_Values[1] - x_Values[0];
    y_spacing   : float         = y_Values[1] - y_Values[0];
    z_spacing   : float         = z_Values[1] - z_Values[0];

    V           : float         = x_spacing*y_spacing*z_spacing;

    # Next, set up the weight function.
    r   : int               = 1;
    X_0 : numpy.ndarray     = torch.tensor([0,0,0], dtype = torch.float32);
    w : Weight_Function     = Weight_Function(X_0 = X_0, r = r, Coords = Coords);

    # Set up a Derivative operator.
    Encoding : numpy.ndarray = numpy.array([1, 2,0], dtype = numpy.int32);
    D : Derivative = Derivative(Encoding = Encoding);

    # Calculate D(w).
    w.Add_Derivative(D = D);

    # We will integrate D(w) over Coords. For this, we need an array of ones.
    FU_Grid : torch.Tensor = torch.empty([Gridlines_per_axis, Gridlines_per_axis, Gridlines_per_axis], dtype = torch.float32);
    for i in range(Gridlines_per_axis):
        for j in range(Gridlines_per_axis):
            for k in range(Gridlines_per_axis):
                FU_Grid[i, j, k] = (x_Values[i].item()*x_Values[i].item() +
                                    y_Values[j].item()*y_Values[j].item() +
                                    z_Values[k].item()*z_Values[k].item());

    FU_Grid = FU_Grid.view(-1);

    print("Done!");

    # Integrate!
    print("Integrating... ", end = '');
    timer : float = time.perf_counter();
    Integral : float        = Integrate(w = w, D = D, FU_Grid = FU_Grid, V = V).item();
    print("Done! Took %fs" % (time.perf_counter() - timer));
    print(Integral);



if __name__ == "__main__":
    Test_Integrate();
