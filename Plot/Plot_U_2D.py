# Nonsense to add Code diectory to the Python search path.
import os
import sys

# Get path to parent directory
parent_dir  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
Code_path   = os.path.join(parent_dir, "Code");
sys.path.append(Code_path);

# Code files.
from Network        import Neural_Network;

import torch;
import numpy;
import matplotlib.pyplot as pyplot;



def Plot_U_2D(  Num_Hidden_Layers   : int,
                Units_Per_Layer     : int,
                Activation_Function : str,
                Device              : torch.device,
                Load_File_Name      : str,
                Gridlines_Per_Axis  : int,
                t_low               : float,
                t_high              : float,
                x_low               : float,
                x_high              : float) -> None:
    """ To do :D

    Note: This can only plot networks with 2 inputs. """

    # First, set up the network.
    U = Neural_Network(
            Num_Hidden_Layers   = Num_Hidden_Layers,
            Neurons_Per_Layer   = Units_Per_Layer,
            Input_Dim           = 2,
            Output_Dim          = 1,
            Activation_Function = Activation_Function,
            Device              = Device);

    # Next, Load U.
    Load_File_Path : str = "../Saves/" + Load_File_Name;
    Saved_State = torch.load(Load_File_Path, map_location = Device);
    U.load_state_dict(Saved_State["U"]);

    # Now, set up the grid ofs coordinates.
    t_Values : numpy.ndarray    = numpy.linspace(start = t_low, stop = t_high, num = Gridlines_Per_Axis);
    x_Values : numpy.ndarray    = numpy.linspace(start = x_low, stop = x_high, num = Gridlines_Per_Axis);

    t_Coords_Matrix : numpy.ndarray = numpy.empty(shape = [Gridlines_Per_Axis, Gridlines_Per_Axis], dtype = numpy.float32);
    x_Coords_Matrix : numpy.ndarray = numpy.empty(shape = [Gridlines_Per_Axis, Gridlines_Per_Axis], dtype = numpy.float32);

    for i in range(Gridlines_Per_Axis):
        t_Coords_Matrix[i, :] = t_Values[i];
        x_Coords_Matrix[:, i] = x_Values[i];

    t_Coords_1D : numpy.ndarray = t_Coords_Matrix.flatten().reshape(-1, 1);
    x_Coords_1D : numpy.ndarray = x_Coords_Matrix.flatten().reshape(-1, 1);

    # Generate data coordinates, corresponding Data Values.
    Coords      : numpy.ndarray = numpy.hstack((t_Coords_1D, x_Coords_1D));

    # Evaluate the network at these coordinates.
    U_Coords    : torch.Tensor  = U(torch.from_numpy(Coords)).view(-1);
    U_matrix    : numpy.ndarray = (U_Coords.detach().numpy()).reshape(Gridlines_Per_Axis, Gridlines_Per_Axis);

    # Plot!!!
    epsilon : float = .0001;
    U_min : float = numpy.min(U_matrix) - epsilon;
    U_max : float = numpy.max(U_matrix) + epsilon;

    # Plot!
    pyplot.contourf(    t_Coords_Matrix,
                        x_Coords_Matrix,
                        U_matrix,
                        levels      = numpy.linspace(U_min, U_max, 500),
                        cmap        = pyplot.cm.jet);

    pyplot.colorbar();
    pyplot.xlabel("t");
    pyplot.ylabel("x");
    pyplot.show();


if __name__ == "__main__":

    # Set the settings.
    Num_Hidden_Layers       : int           = 5;
    Units_Per_Layer         : int           = 50;
    Activation_Function     : str           = "Tanh";
    Device                  : torch.device  = torch.device('cpu');
    Load_File_Name          : str           = "Test_LBFGS";
    Gridlines_Per_Axis      : int           = 200;
    t_low                   : float         = 0.0;
    t_high                  : float         = 10.0;
    x_low                   : float         = -8.0;
    x_high                  : float         = 8.0;

    # Plot!
    Plot_U_2D(  Num_Hidden_Layers   = Num_Hidden_Layers,
                Units_Per_Layer     = Units_Per_Layer,
                Activation_Function = Activation_Function,
                Device              = Device,
                Load_File_Name      = Load_File_Name,
                Gridlines_Per_Axis  = Gridlines_Per_Axis,
                t_low               = t_low,
                t_high              = t_high,
                x_low               = x_low,
                x_high              = x_high);
