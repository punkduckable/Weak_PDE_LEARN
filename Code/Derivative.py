import numpy;



class Derivative():
    """ Objects of this class house an abstract representation of a partial
    derivative operator.

    ----------------------------------------------------------------------------
    Members:

    Encodoing: A 1D numpy array of integers characterizing the partial
    derivative operator. If there are n spatial variables, then this should be a
    n + 1 element array, whose 0 element holds the number of time derivatives,
    and whose ith element (for i > 0) holds the derivative order with respect to
    the i-1th spatial variable. Currently, we only support n = 1, 2, 3.

    Num_Spatial_Vars: The number of spatial variables that a function must
    depend on to be in the domain of the encoded partial derivative operator.
    This is just one less than the number of elements in Encoding. We use this
    mainly for book-keeping purposes. This must be 1, 2, or 3. If it is 2, for
    example, then the encoded partial derivative operator contains partial
    derivatives with respect to x and y, which means that it can only be applied
    to functions of at least two spatial variables. """


    def __init__(   self,
                    Encoding : numpy.ndarray) -> None:
        """ Initializer.

        ------------------------------------------------------------------------
        Arguments:

        Encodoing: See class docstring. """

        # First, cast to integer array. This also returns a copy of Encoding.
        Encoding : numpy.ndarray = Encoding.astype(dtype = numpy.int32);

        # check that Encoding is a 1D array.
        assert(len(Encoding.shape) == 1);

        # Determine number of spatial variables. Currently, we only support
        # one two or three spatial variables.
        Num_Spatial_Vars : int  = Encoding.size - 1;
        assert(Num_Spatial_Vars == 1 or Num_Spatial_Vars == 2 or Num_Spatial_Vars == 3);
        self.Num_Spatial_Vars   = Num_Spatial_Vars;

        # Check that each element of encoding is a non-negative integer.
        for i in range(Encoding.size):
            assert(Encoding[i] >= 0);

        # Assuming the input passes all checks, assign it.
        self.Encoding = Encoding;



    def __str__(self) -> str:
        """ This function returns a string that contains a human-readable
        expression for the derivative operator that this object represents. It
        is mainly used for printing. """

        Buffer : str = "";

        # Time derivative.
        if  (self.Encoding[0] == 1):
            Buffer += "D_t ";
        elif(self.Encoding[0] > 1 ):
            Buffer += ("D_t^%u " % self.Encoding[0]);

        # x derivative.
        if  (self.Encoding[1] == 1):
            Buffer += "D_x ";
        elif(self.Encoding[1] > 1 ):
            Buffer += ("D_x^%u " % self.Encoding[1]);

        # y derivative (if it exists)
        if(self.Num_Spatial_Vars > 1):
            if  (self.Encoding[2] == 1):
                Buffer += "D_y ";
            elif(self.Encoding[2] > 1 ):
                Buffer += ("D_y^%u " % self.Encoding[2]);

        # z derivative (if it exists).
        if(self.Num_Spatial_Vars > 2):
            if  (self.Encoding[3] == 1):
                Buffer += "D_z ";
            elif(self.Encoding[3] > 1 ):
                Buffer += ("D_z^%u " % self.Encoding[3]);

        return Buffer;
