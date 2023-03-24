# Nonsense to add Code directory to the Python search path.
import os
import sys

# Get path to main directory
Main_Path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Get the path to the Code, Classes directories.
Code_Path       = os.path.join(Main_Path, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add these directories to the python search path.
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# external libraries and stuff.
import numpy;
import torch;
import math;
import matplotlib.pyplot as pyplot;

# Code files.
from Derivative             import Derivative;
from Weight_Function        import Weight_Function, Build_From_Other;



def Plot_Bump(  w       : Weight_Function,
                Dim_0   : int,
                Dim_1   : int,
                m       : int,
                Coords  : torch.Tensor) -> None:
    """ 
    This function plots a slice of the Weight Function, w. If w is a
    function of two variables (t and x), then this simply plots w. Otherwise,
    it plots a slice of w. To see this, suppose that Dim_0 = 0, Dim_1 = 1 and
    w is defined on R^4. Then, if w is centered about X_0, we plot w restricted
    to the domain [a, b] x [c, d] x X_0[3] x X_0[4], for some a, b, c, d. 
    """

    # First, get the center and radius of w.
    Center : torch.Tensor   = w.X_0;
    Radius : float          = w.r;

    # Evaluate w at the coords.
    w_Coords : numpy.ndarray = w(Coords).detach().numpy();

    # Set up plotting coordinates.
    grid_Dim0_Coords : numpy.ndarray = Coords[:, Dim_0].reshape(m, m);
    grid_Dim1_Coords : numpy.ndarray = Coords[:, Dim_1].reshape(m, m);
    grid_w_Coords    : numpy.ndarray = w_Coords.reshape(m, m);

    # Get min and max of w_Coords. We will use this to set up colors plot colors.
    epsilon         : float = .0001;
    w_Coords_min    : float = numpy.min(grid_w_Coords) - epsilon;
    w_Coords_max    : float = numpy.max(grid_w_Coords) + epsilon;

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



def Plot_Bump_Derivative(   w                   : Weight_Function,
                            D                   : Derivative,
                            Dim_0               : int,
                            Dim_1               : int,
                            m                   : int,
                            Coords              : torch.Tensor, 
                            Supported_Indices   : torch.Tensor) -> None:
    # First, get the center and radius of w.
    Center : torch.Tensor   = w.X_0;
    Radius : float          = w.r;

    # Evaluate Derivative of w at the coords.
    D_w_Supported   : numpy.ndarray = w.Derivatives[tuple(D.Encoding)].detach().numpy();
    Num_Coords      : int           = Coords.shape[0];

    D_w             : torch.Tensor  = numpy.zeros(Num_Coords, dtype = numpy.float32);
    D_w[Supported_Indices]          = D_w_Supported;

    # Set up plotting coordinates.
    grid_Dim0_Coords : numpy.ndarray = Coords[:, Dim_0].reshape(m, m);
    grid_Dim1_Coords : numpy.ndarray = Coords[:, Dim_1].reshape(m, m);
    grid_D_w         : numpy.ndarray = D_w.reshape(m, m);

    # Get min and max of D_w. We will use this to set up colors plot colors.
    epsilon : float = .0001;
    D_w_min : float = numpy.min(grid_D_w) - epsilon;
    D_w_max : float = numpy.max(grid_D_w) + epsilon;

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
    # Set center, radius of weight function.
    Center : torch.Tensor   = torch.tensor([0, 0], dtype = torch.float32);
    Radius : float          = 3;


    ############################################################################
    # Define the grid we will evaluate the weight function on.

    # Since we can only plot a function of two variables, the coords will lie
    # along a 2D plane... first, select which two dimensions.
    Dim_0 : int = 0
    Dim_1 : int = 1;

    # The coordinates will lie in a Box [a, b] x [c, d] centered at Center, with
    # side length 3r, in the Dim_0 x Dim_1 plane.
    a : float = -3.0*Radius;
    b : float = 3.0*Radius;

    c : float = -3.0*Radius;
    d : float = 3.0*Radius;

    # Place a grid on this box with m grid-lines along each axis.
    m : int = 300;
    Dim0_Pts : numpy.ndarray = numpy.linspace(start = a, stop = b, num = m);
    Dim1_Pts : numpy.ndarray = numpy.linspace(start = c, stop = d, num = m);

    # Generate Coords (non Dim_0/Dim_1 components are the corresponding
    # components of Center)
    n : int = torch.numel(Center);
    Coords  = torch.empty(size = (m*m, n), dtype = torch.float32);

    for i in range(m):
        for j in range(m):
            Coords[m*i + j, :]     = Center;

            Coords[m*i + j, Dim_0] = Dim0_Pts[i];
            Coords[m*i + j, Dim_1] = Dim1_Pts[j];

    # Find the volume for the ith data set (and report).
    V : float = (b - a)*(d - c)/(m*m);

    

    ############################################################################
    # Initialize Derivative operator.

    D = Derivative(Encoding = numpy.array([2, 2], dtype = numpy.int32));


    ############################################################################
    # Initialize weight function

    W_0 : Weight_Function = Weight_Function(X_0     = Center,
                                            r       = Radius,
                                            Coords  = Coords, 
                                            V       = V);
    
    # Find set of supported coordinates. 
    XmX0                        : torch.Tensor  = torch.subtract(Coords, Center);
    Max_XmX0                    : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);
    Supported_Indices           : torch.Tensor  = torch.less(Max_XmX0, Radius);

    # Now add the derivative
    W_0.Add_Derivative(D = D);

    # Now build a second bump from the first. 
    W_1 = Build_From_Other(
                X_1 = torch.tensor([Radius, Radius], dtype = torch.float32), 
                r_1 = Radius*1.5, 
                W_0 = W_0);

    X_0 : torch.Tensor  = W_0.X_0;
    X_1 : torch.Tensor  = W_1.X_0;
    r_0 : float         = W_0.r;
    r_1 : float         = W_0.r;

    W_1_Coords : torch.Tensor = (Coords - X_0)*(r_1/r_0) + X_1;


    ############################################################################
    # Plot!!!

    Plot_Bump(              W_0,      Dim_0 = Dim_0, Dim_1 = Dim_1, m = m, Coords = Coords);
    Plot_Bump_Derivative(   W_0,  D,  Dim_0 = Dim_0, Dim_1 = Dim_1, m = m, Coords = Coords, Supported_Indices = Supported_Indices);

    Plot_Bump(              W_1,      Dim_0 = Dim_0, Dim_1 = Dim_1, m = m, Coords = Coords);
    Plot_Bump_Derivative(   W_1,  D,  Dim_0 = Dim_0, Dim_1 = Dim_1, m = m, Coords = W_1_Coords, Supported_Indices = Supported_Indices);



if __name__ == "__main__":
    main();
