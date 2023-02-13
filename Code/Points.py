import numpy;
import torch;
import random;



def Generate_Points(
        Bounds     : numpy.array,
        Num_Points : int,
        Device     : torch.device = torch.device('cpu')) -> torch.Tensor:
    """ 
    This function generates a two-dimensional tensor, each row of which holds a 
    randomly generated coordinate that lies in the rectangle defined by Bounds.

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
    within [a_1, b_1]x...x[a_n, b_n]. 
    """


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



def Setup_Partition(Axis_Partition_Size : int,
                    Bounds              : numpy.ndarray) -> torch.Tensor:
    """ 
    This function sets up a uniform partition with Axis_Partition_Size points 
    along each axis. Suppose Bounds = [[a_1, b_1], ... ,[a_n, b_n]]. We
    partition [a_i, b_i] using Axis_Partition_Size points such that successive
    points in the partition are equally spaced ( with a spacing of
    (b_i - a_i)/(Axis_Partition_Size - 1)). We repeat this for each dimension,
    which yields a partition of [a_1, b_1] x ... x [a_n, b_n]. In particular,
    this is the set of all points of the form (p_1, ... , p_n) where each
    p_i is an element of the partition of [a_i, b_i]. Currently, this function
    works when n = 2, 3, and 4.

    ----------------------------------------------------------------------------
    Arguments:

    Axis_Partition_Size : The number of gird lines we want along each axis. This
    is the size of the partition of any sub-rectangle, [a_i, b_i].

    Bounds : A 2D array whose ith row is [a_i, b_i]. This function generates a
    uniform partition on the rectangle [a_1, b_1] x ... x [a_n, b_n]. Bounds
    defines the rectangle.

    ----------------------------------------------------------------------------
    Returns:

    A 2D array whose ith row holds the coordinates of the ith partition point.
    In particular, this array will have size (Axis_Partition_Size^n) x n. """

    # Aliases.
    Nd : int = Bounds.shape[0];
    Ng : int = Axis_Partition_Size;

    # For readability, we handle the cases of 1, 2, and 3 spatial variables
    # separately.
    if  (Nd == 2):
        # Initialize a tensor to holds the partition coordinates. We will return
        # this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Axis_Partition_Size):
            Coords[i, :, 0] = t_Values[i].item();
            Coords[:, i, 1] = x_Values[i].item();

        return Coords.view(-1, Nd);

    elif(Nd == 3):
        # Initialize a tensor to holds the partition coordinates. We will return
        # this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);
        y_Values : numpy.ndarray = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Ng):
            Coords[i, :, :, 0] = t_Values[i].item();
            Coords[:, i, :, 1] = x_Values[i].item();
            Coords[:, :, i, 2] = y_Values[i].item();

        return Coords.view(-1, Nd);

    elif(Nd == 4):
        # Initialize a tensor to holds the partition coordinates. We will return
        # this after setting it up.
        Coords : torch.tensor = torch.empty([Ng, Ng, Ng, Ng, Nd], dtype = torch.float32);

        # Determine possible t, x values.
        t_Values : numpy.ndarray = numpy.linspace(start = Bounds[0, 0], stop = Bounds[0, 1], num = Ng, dtype = numpy.float32);
        x_Values : numpy.ndarray = numpy.linspace(start = Bounds[1, 0], stop = Bounds[1, 1], num = Ng, dtype = numpy.float32);
        y_Values : numpy.ndarray = numpy.linspace(start = Bounds[2, 0], stop = Bounds[2, 1], num = Ng, dtype = numpy.float32);
        z_Values : numpy.ndarray = numpy.linspace(start = Bounds[3, 0], stop = Bounds[3, 1], num = Ng, dtype = numpy.float32);

        # Populate Coords.
        for i in range(Ng):
            Coords[i, :, :, :, 0] = t_Values[i].item();
            Coords[:, i, :, :, 1] = x_Values[i].item();
            Coords[:, :, i, :, 2] = y_Values[i].item();
            Coords[:, :, :, i, 3] = z_Values[i].item();

        return Coords.view(-1, Nd);

    else:
        print("Invalid number of dimensions. Must be 2, 3, or 4. Got %d" % Nd);
        exit();
