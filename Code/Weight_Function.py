import torch;




class Weight_Function(torch.nn.Module):
    """ Objects of this class are bump functions with a specified radius, X_0,
    and radius, r. In particular, each object of this class represents the
    function w : R^n -> R defined by
            w(X) =  { exp(-1/(1 - (||X - X_0||/r)^2))     if ||X - X_0|| < r
                    { 0                                   otherwise
    Such a function is infinitely differentiable on R^n.

    ----------------------------------------------------------------------------
    Members:

    X_0 : The center of the bump function, as defined above. This should be a
    1D tensor whose length is n (the dimension of the domain of w).

    r : the radius of the bump function. The bump function is zero outside of
    B_r(X_0).

    Input_Dim : w (see above) is defined on R^n. This is n.

    Derivatives : A dictionary. For this dictionary, an elements key is a
    Derivative object's encoding array. Its value is a torch tensor that holds
    the value of the corresponding derivative of the weight function at a
    set of coordinates.  """

    def __init__(self, X_0 : torch.Tensor, r : float) -> None:
        """ Class initializer.

        ------------------------------------------------------------------------
        Arguments:

        X_0 : The center of the weight function. See class docstring.

        r : The radius of the weight function. See class docstring. """

        # First, call the module initializer.
        super(Weight_Function, self).__init__();

        # Next, check that X_0 is a 1D array.
        assert(len(X_0.shape) == 1);

        # Next, check that r > 0.
        assert(r > 0);

        # Assuming we passed the checks, assign X_0 and r members. We can also
        # determine the input dimension from the length of X_0.
        self.X_0        : torch.Tensor  = X_0;
        self.Input_Dim  : int           = X_0.numel();
        self.r          : float         = r;

        # Finally, intialize this object's derivatives array.
        self.Derivatives : dict = {};

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ This function evaluates the weight function at each coordinate of X.

        ------------------------------------------------------------------------
        Arguments:

        X : A 2D tensor, each row of which holds a coordinate at which we want
        to evaluate the weight function. If this weight function object is
        defined on R^n, then X should be a B by n tensor.

        ------------------------------------------------------------------------
        Returns:

        A B element tensor whose ith entry is w evaluated at the ith coordinate
        (ith row of X). """

        # Recall that w(X) = { exp( 1/((||X - X_0||/r)^2 - 1) )    if || X - X_0|| < r.
        #                    { 0                                     otherwise

        # First, calculate || X - X_0 ||_2^2.
        XmX0                : torch.Tensor  = torch.subtract(X, self.X_0);
        Norm_XmX0           : torch.Tensor  = torch.sum(torch.multiply(XmX0, XmX0), dim = 1);

        # Now, determine which points are in B_r(X_0), and which are not.
        Indices_In_BrX0     : torch.Tensor  = torch.less(Norm_XmX0, (self.r)*(self.r));

        # Evaluate w at the points in B_r(X_0).
        Norm_XmX0_In_BrX0   : torch.Tensor  = Norm_XmX0[Indices_In_BrX0];
        Exp_Denominator     : torch.Tensor  = torch.subtract(torch.divide(Norm_XmX0_In_BrX0, (self.r*self.r)), 1.);
        w_X_In_BrX0         : torch.Tensor  = torch.exp(torch.divide(torch.ones_like(Exp_Denominator), Exp_Denominator));

        # Put everything together.
        w_X                 : torch.Tensor  = torch.zeros_like(Norm_XmX0);
        w_X[Indices_In_BrX0]                = w_X_In_BrX0;

        # Return!
        return w_X;
