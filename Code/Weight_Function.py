import torch;

from Derivative             import Derivative;



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
        """ This method evaluates the weight function at each coordinate of X.

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



    def Add_Derivative(     self,
                            D       : Derivative,
                            Coords  : torch.Tensor) -> None:
        """ Let w denote this weight function object. This method evaluates
        D(w) at Coords, and then stores the result in w.Derivatives. Critically,
        we only perform these calculations if w.Derivatives does not contain
        an entry corresponding to D. One consequence of this is that once we
        add an entry to w.Derivatives for a particular D, we can not change the
        coordinates at which we evaluated D(w).

        ------------------------------------------------------------------------
        Arguments:

        D : A derivative operator. We evaluate D(w) at Coords. In general, D(w)
        must make sense. This means that if w is a function on R^n, then
        D.Encoding should contain at most n elements. Why? D.Encoding[k]
        represents the number of derivatives with respect to the kth variable.
        This really only makes sense if w depends on at least k variables.

        Coords : The coordinates at which we evaluate D(w). If w is defined on
        R^n, then this should be a B by n tensor whose ith row is the ith
        coordinate at which we want to evaluate D(w).

        ------------------------------------------------------------------------
        Returns:

        A 1D tensor. If Coords has B rows, then this is a B element tensor
        whose ith row holds the value of D(w) at the ith coordinate (ith row of
        Coords). """

        # First, check if we've already evaluated D(w).
        if(tuple(D.Encoding) in self.Derivatives):
            print(D + "w already in Derivatives dictionary.");
            return;

        # If not, then let's calculate it.
        Dw : torch.Tensor = Evaluate_Derivative(    w       = self,
                                                    D       = D,
                                                    Coords  = Coords);
        self.Derivatives[tuple(D.Encoding)] = Dw;

        # All done!
        return;



    def Get_Derivative(self, D : Derivative):
        """ This function returns the element of self.Derivatives corresponding
        to D(w) (assuming it exists). That entry of self.Derivatives should hold
        D(w) evaluated at a set of coordinates. See Add_Derivative.

        ------------------------------------------------------------------------
        Arguments:

        D : A derivative operator. We fetch the element of self.Derivatives
        corresponding to D.

        ------------------------------------------------------------------------
        Returns:

        The entry of self.Derivatives corresponding to D. This should be an
        1D array whose entries are D(w) at some set of coordinates. """

        return self.Derivatives[tuple(D.Encoding)];



def Evaluate_Derivative(
        w           : Weight_Function,
        D           : Derivative,
        Coords      : torch.Tensor) -> torch.Tensor:
    """ This function applies a particular partial derivative operator, D, of
    a weight function, w, and then evaluates the resulting function on Coords.
    In other words, this function evaluates D(w)(X), for each X in Coords.

    ----------------------------------------------------------------------------
    Arguments:

    w : The weight function whose derivative we evaluate. In general, w must
    depend on "enough" variables for D(w) to make sense. In other words, if
    w is defined on R^n, then D's Encoding vector must have <= n elements. Why?
    Recall that the kth component of D's encoding vector represents the partial
    derivative order with respect to the kth variable. For this to make sense,
    w must be a function of at least k variables.

    D : A derivative object. We apply the partial derivative operator that
    D represents to w.

    Coords : The points at which we evaluate D(w). Is w is defined on R^n, then
    this should be an B by n tensor whose ith row holds the ith coordinate at
    which we want to evaluate D(w). Thus, each row of D(w) represents a point in
    R^n.

    ----------------------------------------------------------------------------
    Returns:

    This returns a 1D Tensor. If Coords has B rows, then the returned tensor has
    B elements. The ith component of the returned Tensor holds D(w) evaluated
    at the ith coordinate (ith row of Coords). """

    # We need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # Make sure we can actually compute the derivatives. For this, we need
    # the input dimension of f to be <= the size of D's encoding vector.
    assert(D.Encoding.size <= w.Input_Dim);

    # Now, let's get to work. The plan is the following: Suppose we want to find
    # D_t^{m(t)} D_x^{m(x)} D_y^{m(y)} D_z^{m(z)} w. First, we compute
    # D_t^{m(t)} w. From this, we calculuate D_x^{m(x)} D_t^{m(t)} w, and so on.
    # We then use equality of mixed partials (remember, w is infinitely
    # differentiable) to rewrite this as the derivative we want.
    w_Coords : torch.Tensor = w(Coords).view(-1);



    ############################################################################
    # t derivatives.

    # Initialize Dt_w. If there are no t derivatives, then Dt_w = w_Coords.
    Dt_w : torch.Tensor = w_Coords;

    Dt_Order : int = D.Encoding[0];
    if(Dt_Order > 0):
        # Suppose Dt_Order = m. Compute D_t^k w from D_t^{k - 1} w for each k
        # in {1, 2, ... , m}.
        for k in range(1, Dt_Order + 1):
            # Compute the gradient.
            Grad_Dt_w : torch.Tensor = torch.autograd.grad(
                            outputs         = Dt_w,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dt_w),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Update Dt_w (this replaces D_t^{k - 1} w with D_t^k w)
            Dt_w = Grad_Dt_w[:, 0].view(-1);



    ############################################################################
    # x derivatives.

    # Initialize Dx_Dt_w. If there are no x derivatives, then Dx_Dt_w = Dx_w.
    Dx_Dt_w : torch.Tensor = Dt_w;

    Dx_Order : int = D.Encoding[1];
    if(Dx_Order > 0):
        # Suppose Dx_Order = m. We compute D_x^k Dt_w from D_t^{k - 1} Dt_w for
        # each k in {1, 2, ... , m}.
        for k in range(1, Dx_Order + 1):
            # Compute the gradient.
            Grad_Dx_Dt_w : torch.Tensor = torch.autograd.grad(
                            outputs         = Dx_Dt_w,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dx_Dt_w),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Update Dx_Dt_w (this replaces D_x^{k - 1} Dt_w with D_x^k Dt_w)
            Dx_Dt_w = Grad_Dx_Dt_w[:, 1].view(-1);



    ############################################################################
    # y derivatives.

    # First, check if there are any y derivatives (if Derivative.Encoding has a
    # 3rd element). If not, then we're done.
    if(D.Encoding.size < 3):
        return Dx_Dt_w;

    # Assuming we need y derivatives, initialize Dy_Dx_Dt_w. If there are no y
    # derivatives, then Dy_Dx_Dt_w = Dx_Dt_w.
    Dy_Dx_Dt_w : torch.Tensor = Dx_Dt_w;

    Dy_Order : int = D.Encoding[2];
    if(Dy_Order > 0):
        # Suppose Dy_Order = m. We compute D_y^k Dx_Dt_w from
        # D_y^{k - 1} Dx_Dt_w for each k in {1, 2, ... , m}.
        for k in range(1, Dy_Order + 1):
            # Compute the gradient.
            Grad_Dy_Dx_Dt_w : torch.Tensor = torch.autograd.grad(
                            outputs         = Dy_Dx_Dt_w,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dy_Dx_Dt_w),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Update Dy_Dx_Dt_w (this replaces D_y^{k - 1} Dx_Dt_w with
            # D_y^k Dx_Dt_w)
            Dy_Dx_Dt_w = Grad_Dy_Dx_Dt_w[:, 2].view(-1);



    ############################################################################
    # z derivatives.

    # First, check if there are any z derivatives (if Derivative.Encoding has a
    # 4th element). If not, then we're done.
    if(D.Encoding.size < 4):
        return Dy_Dx_Dt_w;

    # Assuming we need z derivatives, initialize Dz_Dy_Dx_Dt_w. If there are no
    # z derivatives, then Dz_Dy_Dx_Dt_w = Dy_Dx_Dt_w.
    Dz_Dy_Dx_Dt_w : torch.Tensor = Dy_Dx_Dt_w;

    Dz_Order : int = D.Encoding[3];
    if(Dz_Order > 0):
        # Suppose Dz_Order = m. We compute D_z^k Dy_Dx_Dt_w from
        # D_z^{k - 1} Dy_Dx_Dt_w for each k in {1, 2, ... , m}.
        for k in range(1, Dz_Order + 1):
            # Compute the gradient.
            Grad_Dz_Dy_Dx_Dt_w : torch.Tensor = torch.autograd.grad(
                            outputs         = Dz_Dy_Dx_Dt_w,
                            inputs          = Coords,
                            grad_outputs    = torch.ones_like(Dz_Dy_Dx_Dt_w),
                            retain_graph    = True,
                            create_graph    = True)[0];

            # Update Dz_Dy_Dx_Dt_w (this replaces D_y^{k - 1} Dy_Dx_Dt_w with
            # D_y^k Dy_Dx_Dt_w)
            Dz_Dy_Dx_Dt_w = Grad_Dz_Dy_Dx_Dt_w[:, 3].view(-1);

    return Dz_Dy_Dx_Dt_w;
