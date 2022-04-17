import numpy;
import torch;
import random;



def Generate_Points(
        Bounds     : numpy.array,
        Num_Points : int,
        Device     : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ This function generates a two-dimensional tensor, each row of which
    holds a randomly generated coordinate that lies in the rectangle defined by
    Bounds.

    ----------------------------------------------------------------------------
    Arguments:

    Bounds: A two-column tensor. Whose ith row contains the lower and upper
    bounds of the ith sub-rectangle of the rectangle.

    Num_Points: The number of points we want to generate.

    Device: The device you want the Point tensor to be stored on.

    ----------------------------------------------------------------------------
    Returns:

    A Num_Points row tensor, each row of which contains a randomly generated
    coordinate in the rectangle specified by Bounds. Suppose that
            Bounds = [[a_1, b_1], ... , [a_n, b_n]]
    Then the ith row of the returned tensor contains a coordinate that lies
    within [a_1, b_1]x...x[a_n, b_n]. """


    # First, determine the number of dimensions. This is just the number of rows
    # in Bounds.
    Num_Dim : int = Bounds.shape[0];

    # Check that the Bounds are valid.
    for j in range(Num_Dim):
        assert(Bounds[j, 0] <= Bounds[j, 1]);

    # Make a tensor to hold all the points.
    Points = torch.empty((Num_Points, Num_Dim),
                          dtype  = torch.float32,
                          device = Device);

    # Populate the coordinates in Points, one coordinate at a time.
    for j in range(Num_Dim):
        # Get the upper and lower bounds for the jth coordinate.
        Lower_Bound : float = Bounds[j, 0];
        Upper_Bound : float = Bounds[j, 1];

        # Cycle through the points.
        for i in range(Num_Points):
            Points[i, j] = random.uniform(Lower_Bound, Upper_Bound);

    return Points;



def Setup_Grid( Gridlines_Per_Axis  : int,
                Num_Dimensions      : int,
                Bounds              : numpy.ndarray) -> torch.Tensor:
    """ To do :D """

    # Alies.
    Nd : int = Num_Dimensions
    Ng : int = Gridlines_Per_Axis;

    # For readability, we handle the cases of 1, 2, and 3 spatial variables
    # separately.
    if  (Nd == 2):
        # Initialize Grids_Coords. We will return this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Gridlines_Per_Axis):
            Coords[i, :, 0] = t_Values[i].item();

            for j in range(Gridlines_Per_Axis):
                Coords[i, j, 1] = x_Values[j].item();

        return Coords.view(-1, Nd);

    elif(Nd == 3):
        # Initialize Grids_Coords. We will return this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);
        y_Values : numpy.ndarray = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Ng):
            Coords[i, :, :, 0] = t_Values[i].item();

            for j in range(Ng):
                Coords[i, j, :, 1] = x_Values[j].item();

                for k in range(Ng):
                    Coords[i, j, k, 2] = y_Values[k].item();


        return Coords.view(-1, Nd);

    elif(Nd == 4):
        # Initialize Grids_Coords. We will return this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);
        y_Values : numpy.ndarray = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Ng, dtype = numpy.float32);
        z_Values : numpy.ndarray = numpy.linspace(start = Bounds[3, 0], stop = Bounds[3, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Ng):
            Coords[i, :, :, :, 0] = t_Values[i].item();

            for j in range(Ng):
                Coords[i, j, :, :, 1] = x_Values[j].item();

                for k in range(Ng):
                    Coords[i, j, k, :, 2] = y_Values[k].item();

                    for l in range(Ng):
                        Coords[i, j, k, l, 3] = z_Values[l].item();

        return Coords.view(-1, Nd);

    else:
        print("Invalid number of dimensions. Must be 2, 3, or 4. Got %d" % Nd);
        exit();
