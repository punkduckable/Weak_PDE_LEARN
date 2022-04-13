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
import math;
import matplotlib.pyplot as pyplot;

# Code files.
from Derivative             import Derivative;
from Weight_Function        import Weight_Function;
from Evaluate_Derivative    import Evaluate_Derivative;



def Plot_Bump(  w       : Weight_Function,
                Dim_0   : int,
                Dim_1   : int) -> None:
    """ This function plots a slice of the Weight Function, w. If w is a
    function of two variables (t and x), then this simply plots w. Otherwise,
    it plots a slice of w. To see this, suppose that Dim_0 = 0, Dim_1 = 1 and
    w is defined on R^4. Then, if w is centered about X_0, we plot w restriced
    to the domain [a, b] x [c, d] x X_0[3] x X_0[4], for some a, b, c, d. """

    # First, get the center and radius of w.
    Center : torch.Tensor   = w.X_0;
    Radius : float          = w.r;

    # Make a Box, [t_low, t_high] x [x_low, x_high] centered at Center, with
    # side length 3r.
    a : float = Center[Dim_0] - 1.5*Radius;
    b : float = Center[Dim_0] + 1.5*Radius;

    c : float = Center[Dim_1] - 1.5*Radius;
    d : float = Center[Dim_1] + 1.5*Radius;

    # Place a grid on this box with 100 gridlines along each axis.
    Dim0_Pts : numpy.ndarray = numpy.linspace(start = a, stop = b, num = 100);
    Dim1_Pts : numpy.ndarray = numpy.linspace(start = c, stop = d, num = 100);

    # Generate coords.
    n : int = torch.numel(Center);
    Coords  = torch.empty(size = (100*100, n), dtype = torch.float32);

    for i in range(100):
        for j in range(100):
            Coords[100*i + j, :] = Center;

            Coords[100*i + j, 0] = Dim0_Pts[i];
            Coords[100*i + j, 1] = Dim1_Pts[j];


    # Evaluate w at the coords.
    w_Coords : numpy.ndarray = w(Coords).detach().numpy();

    # Set up plotting coordinates.
    grid_Dim0_Coords : numpy.ndarray = numpy.empty(shape = (100, 100), dtype = numpy.float32);
    grid_Dim1_Coords : numpy.ndarray = numpy.empty(shape = (100, 100), dtype = numpy.float32);
    grid_w_Coords    : numpy.ndarray = w_Coords.reshape(100, 100);

    for i in range(100):
        for j in range(100):
            grid_Dim0_Coords[i, j]  = Dim0_Pts[i];
            grid_Dim1_Coords[i, j]  = Dim1_Pts[j];

    # Get min and max of w_Coords. We will use this to set up colors plot colors.
    w_Coords_min : float = numpy.min(grid_w_Coords);
    w_Coords_max : float = numpy.max(grid_w_Coords);

    # Plot!
    pyplot.contourf(    grid_Dim0_Coords,
                        grid_Dim1_Coords,
                        grid_w_Coords,
                        levels      = numpy.linspace(w_Coords_min, w_Coords_max, 500),
                        cmap        = pyplot.cm.jet);
    pyplot.colorbar();
    pyplot.xlabel("Dim 0");
    pyplot.ylabel("Dim 1");
    pyplot.show();



def Plot_Bump_Derivative(   w           : Weight_Function,
                            Derivative  : Derivative,
                            Dim_0       : int,
                            Dim_1       : int) -> None:
    # First, get the center and radius of w.
    Center : torch.Tensor   = w.X_0;
    Radius : float          = w.r;

    # Make a Box, [t_low, t_high] x [x_low, x_high] centered at Center, with
    # side length 3r.
    a : float = Center[Dim_0] - 1.5*Radius;
    b : float = Center[Dim_0] + 1.5*Radius;

    c : float = Center[Dim_1] - 1.5*Radius;
    d : float = Center[Dim_1] + 1.5*Radius;

    # Place a grid on this box with 100 gridlines along each axis.
    Dim0_Pts : numpy.ndarray = numpy.linspace(start = a, stop = b, num = 100);
    Dim1_Pts : numpy.ndarray = numpy.linspace(start = c, stop = d, num = 100);

    # Generate coords.
    n : int = torch.numel(Center);
    Coords  = torch.empty(size = (100*100, n), dtype = torch.float32);

    for i in range(100):
        for j in range(100):
            Coords[100*i + j, :] = Center;

            Coords[100*i + j, Dim_0] = Dim0_Pts[i];
            Coords[100*i + j, Dim_1] = Dim1_Pts[j];


    # Evaluate Derivative of w at the coords.
    D_w : numpy.ndarray = Evaluate_Derivative(
                                w       = w,
                                D       = Derivative,
                                Coords  = Coords).detach().numpy();

    # Set up plotting coordinates.
    grid_Dim0_Coords : numpy.ndarray = numpy.empty(shape = (100, 100), dtype = numpy.float32);
    grid_Dim1_Coords : numpy.ndarray = numpy.empty(shape = (100, 100), dtype = numpy.float32);
    grid_D_w         : numpy.ndarray = D_w.reshape(100, 100);

    for i in range(100):
        for j in range(100):
            grid_Dim0_Coords[i, j]  = Dim0_Pts[i];
            grid_Dim1_Coords[i, j]  = Dim1_Pts[j];

    # Get min and max of D_w. We will use this to set up colors plot colors.
    D_w_min : float = numpy.min(grid_D_w);
    D_w_max : float = numpy.max(grid_D_w);

    # Plot!
    pyplot.contourf(    grid_Dim0_Coords,
                        grid_Dim1_Coords,
                        grid_D_w,
                        levels      = numpy.linspace(D_w_min, D_w_max, 500),
                        cmap        = pyplot.cm.jet);

    pyplot.colorbar();
    pyplot.xlabel("Dim 0");
    pyplot.ylabel("Dim 1");
    pyplot.show();



def main():
    Center : torch.Tensor   = torch.tensor([0, 0, 0, 0], dtype = torch.float32);
    Radius : float          = 3;

    w : Weight_Function = Weight_Function(X_0 = Center, r = Radius);

    D = Derivative(Encoding = numpy.array([0, 0, 1, 1], dtype = numpy.int32));
    Plot_Bump(w, 2, 3);
    Plot_Bump_Derivative(w, D, 2, 3);



if __name__ == "__main__":
    main();
