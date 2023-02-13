import torch;
import numpy;
import math;

from Derivative             import Derivative;



class Weight_Function(torch.nn.Module):
    """ 
    Objects of this class are bump functions with a specified radius, X_0, and 
    radius, r. In particular, each object of this class represents the function 
    w : R^n -> R defined by
        w(X) =  { prod_{i = 1}^{n} exp(5 r^2/((X_i - X_0_i)^2 - r^2) + 5)   if ||X - X_0||_{max} < r
                { 0                                                         otherwise
    where p_1, ... , p_n are positive integers. Such a function is C^infinity

    A weight function is always paired with a set of coordinates. We assume that
    you'll want to compute various integrals of the form
            \int_{S} w(X) D^{(k)} F_k(U(X)) dX),
    for some set S. Here, D^{(k)} is a partial derivative operator, and F_k(U)
    is a trial function. To approximate this integral, we evaluate the integrand
    at a set of points in S. We assume that we use the same points for every
    integral involving a particular weight function. Under this assumption, each
    weight function can be paired with a set of coordinates.

    Using the same set of coordinates for each integral also allows for a key
    performance breakthrough: We only need to evaluate w and its derivatives at
    the coordinates once. Once we have these values, we can then use them each
    time we want to compute an integral involving w (or its derivatives). This
    is why weight function objects have the "Derivatives" dictionary. This
    dictionary stores the various derivatives of w, evaluated at the set of
    coordinates paired with w.

    We can, however, make a critical reduction that boosts performance and
    slashes storage requirements: A bump function and its derivatives are zero
    outside of the ball B_r(X_0) (using the infinity norm). As such, there is no
    need to store the value of w or its derivatives at points outside of
    B_r(X_0). Weight functions take advantage of this. In particular, when you
    initialize a weight function, you do so using that function's set of
    coordinates. We determine which coordinates are in B_r(X_0) and down write
    them (and their indices within the sef of coordinates). When the user asks
    us to calculate a particular derivative of w, we simply evaluate it at the
    coordinates we wrote down. If the user asks for the value of a derivative of
    w, we can use the values we wrote down to reconstruct the derivative on the
    entire set of coordinates.

    ----------------------------------------------------------------------------
    Members:

    X_0 : The center of the bump function, as defined above. This should be a
    1D tensor whose length is n (the dimension of the domain of w).

    r : the radius of the bump function. The bump function is zero outside of
    B_r(X_0).

    Input_Dim : w (see above) is defined on R^n. This is n.

    Num_Coords : The number of rows in the Coords tensor passed to the
    Initializer. We use this to reconstruct various derivatives of w (see
    Get_Derivative).

    Supported_Coords : When we initialize a weight function object, we pass a
    set of coordinates. Supported_Coords is the set of coordinates in that set that
    are also in B_r(X_0).

    Supported_Indices : When we initialize a weight function object, we pass a
    set of coordinates. Supported_Indices holds the indices, within the original
    set of coordinates, of those coordinates which are in B_r(X_0).

    Derivatives : A dictionary that holds various derivatives of w evaluated at
    Supported_Coords. For this dictionary, an elements key is a Derivative
    object's encoding array. Its value is a torch tensor that holds the value of
    the corresponding derivative of the weight function at a set of coordinates. 
    """

    def __init__(   self,
                    X_0     : torch.Tensor,
                    r       : float,
                    Powers  : torch.Tensor,
                    Coords  : torch.Tensor) -> None:
        """ 
        Class initializer.

        ------------------------------------------------------------------------
        Arguments:

        X_0 : The center of the weight function. See class docstring.

        r : The radius of the weight function. See class docstring.

        Powers : A list of integers. We are using Polynomial weight functions,
        then this defines the function (as well as how many derivatives we can
        take). If we are using bump functions (either kind), then this is an
        unused parameter.

        Coords : A list of coordinates. Whenever we apply a derivative to w, we
        evaluate (and store) the derivative at the coordinates in Coords. We
        assume this is a B by n dimensional array whose ith row holds the ith
        coordinate. Here, B is the number of coordinates and n is the number of
        components in each coordinate. 
        """

        # First, call the module initializer.
        super(Weight_Function, self).__init__();


        ########################################################################
        # Checks

        assert(len(X_0.shape) == 1);                # X_0 should be a 1D Tensor.
        assert(len(Coords.shape) == 2);             # Coords should be a 2D Tensor.
        assert(Coords.shape[1] == X_0.shape[0]);    # X_0 should be in the same space as the Coords
        assert(r > 0);                              # radius must be positive.
        assert(torch.sum(Powers <= 0) == 0);        # Each power must be positive (to ensure smoothness)
        assert(Powers.numel() == X_0.numel());      # There should be n powers.


        ########################################################################
        # Assuming we passed the checks, assign X_0 and r members. We can also
        # determine the input dimension from the length of X_0.

        self.X_0            : torch.Tensor  = X_0;
        self.Input_Dim      : int           = X_0.numel();
        self.r              : float         = r;
        self.Powers         : torch.Tensor  = Powers.to(dtype = torch.int32);

        # Get Num_Coords.
        self.Num_Coords     : int           = Coords.shape[0];

        # Intialize this object's derivatives array.
        self.Derivatives    : dict = {};


        ########################################################################
        # Now, determine which coordinates are in B_r(X_0).

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0                    : torch.Tensor  = torch.subtract(Coords, X_0);
        Max_XmX0                : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);

        # Now, determine which coordinates are in B_r(X_0).
        self.Supported_Indices    : torch.Tensor  = torch.less(Max_XmX0, self.r);

        # Record the coordinates that are.
        self.Supported_Coords     : torch.Tensor  = Coords[self.Supported_Indices, :];



    # Polynomial. If using, weight functions need a "Powers" attribute and use
    # inf norm to determine supported points.
    #def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ 
        This method evaluates the weight function at each coordinate of X.

        ------------------------------------------------------------------------
        Arguments:

        X : A 2D tensor, each row of which holds a coordinate at which we want
        to evaluate the weight function. If this weight function object is
        defined on R^n, then X should be a B by n tensor.

        ------------------------------------------------------------------------
        Returns:

        A B element tensor whose ith entry is w evaluated at the ith coordinate
        (ith row of X). """
        """
        # w(X) = { prod_{i = 1}^{n} ((X0_i + r - X_i)(X_i - X0_i + r)/r^2)^p_i if ||X - X_0||_{max} < r
        #        { 0                                                            otherwise

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0                : torch.Tensor  = torch.subtract(X, self.X_0);
        Max_XmX0            : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);

        # Determine which coordinates are in B_r(X_0).
        Supported_Indices   : torch.Tensor  = torch.less(Max_XmX0, self.r);

        # Extract the coordinates in B_r(X_0).
        X_Supported         : torch.Tensor  = X[Supported_Indices, :];

        # Calculate (X - (X0 + r))/r and (X - (X0 - r))/r.
        X0prmX              : torch.Tensor  = torch.subtract(torch.add(self.X_0, self.r), X_Supported);
        XmX0pr              : torch.Tensor  = torch.subtract(X_Supported, torch.subtract(self.X_0, self.r));

        # Now compute their element wise product.
        X0prmX_XmX0pr_r2    : torch.Tensor  = torch.divide(torch.multiply(X0prmX, XmX0pr), (self.r)*(self.r));

        # Raise to the power Powers[i].
        X0prmX_XmX0pr_r2_pi : torch.Tensor  = torch.pow(X0prmX_XmX0pr_r2, self.Powers);

        # Compute the product of the columns of the above tensor. This yields a
        # tensor whose ith entry is w(X_i).
        w_X_Supported       : torch.Tensor  = torch.prod(X0prmX_XmX0pr_r2_pi, dim = 1);

        # Calculate w at the rest of the coordinates.
        w_X                 : torch.Tensor  = torch.zeros(X.shape[0], dtype = X.dtype);
        w_X[Supported_Indices]              = w_X_Supported;

        # Return!
        return w_X;
        """



    # Bump in R^n. If using, make init use 2 norm to determine supported points.
    #def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ 
        This method evaluates the weight function at each coordinate of X.

        ------------------------------------------------------------------------
        Arguments:

        X : A 2D tensor, each row of which holds a coordinate at which we want
        to evaluate the weight function. If this weight function object is
        defined on R^n, then X should be a B by n tensor.

        ------------------------------------------------------------------------
        Returns:

        A B element tensor whose ith entry is w evaluated at the ith coordinate
        (ith row of X). """
        """
        # w(X) = { exp( 7.5r^2/(||X - X_0||^2 - r^2) + 7.5 )    if || X - X_0|| < r.
        #        { 0                                            otherwise

        # First, calculate || X - X_0 ||_2^2.
        XmX0                : torch.Tensor  = torch.subtract(X, self.X_0);
        Norm_XmX0           : torch.Tensor  = torch.sum(torch.multiply(XmX0, XmX0), dim = 1);

        # Now, determine which points are in B_r(X_0), and which are not.
        Indices_In_BrX0     : torch.Tensor  = torch.less(Norm_XmX0, (self.r)*(self.r));

        # Evaluate w at the points in B_r(X_0).
        Norm_XmX0_In_BrX0   : torch.Tensor  = Norm_XmX0[Indices_In_BrX0];
        Denominators        : torch.Tensor  = torch.subtract(Norm_XmX0_In_BrX0, (self.r)**2);
        Exponents           : torch.Tensor  = torch.add(torch.divide(torch.full_like(Denominators, 7.5*(self.r)**2), Denominators), 7.5);
        w_X_In_BrX0         : torch.Tensor  = torch.exp(Exponents);

        # Put everything together.
        w_X                 : torch.Tensor  = torch.zeros_like(Norm_XmX0);
        w_X[Indices_In_BrX0]                = w_X_In_BrX0;

        # Return!
        return w_X;
        """



    # Product of 1D bumps. If using, make init use inf norm to determine
    # supported points.
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ 
        This method evaluates the weight function at each coordinate of X.

        ------------------------------------------------------------------------
        Arguments:

        X : A 2D tensor, each row of which holds a coordinate at which we want
        to evaluate the weight function. If this weight function object is
        defined on R^n, then X should be a B by n tensor.

        ------------------------------------------------------------------------
        Returns:

        A B element tensor whose ith entry is w evaluated at the ith coordinate
        (ith row of X). 
        """

        # w(X) =    { prod_{i = 1}^{n} exp(5*r^2/((X_i - X_0_i)^2 - r^2) + 5)   if ||X - X_0||_{max} < r
        #           { 0                                                         otherwise

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0                : torch.Tensor  = torch.subtract(X, self.X_0);
        Max_XmX0            : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);

        # Determine which coordinates are in B_r(X_0).
        Supported_Indices   : torch.Tensor  = torch.less(Max_XmX0, self.r);

        # Evaluate (X_i - X_0_i)^2 - r^2 for each component of each
        # supported coordinate.
        XmX0                : torch.Tensor  = XmX0[Supported_Indices, :];
        Denominators        : torch.Tensor  = torch.subtract(torch.pow(XmX0, 2.), (self.r)**2);

        # Compute w at the supported coordinates.
        r2_XmX02_r2_5       : torch.Tensor = torch.add(torch.divide(torch.full_like(Denominators, 7.5*(self.r)**2), Denominators), 7.5);
        Exponent            : torch.Tensor = torch.sum(r2_XmX02_r2_5, dim = 1);
        w_X_Supported       : torch.Tensor = torch.exp(Exponent);

        # Calculate w at the rest of the coordinates.
        w_X                 : torch.Tensor  = torch.zeros(X.shape[0], dtype = X.dtype);
        w_X[Supported_Indices]              = w_X_Supported;

        # Return!
        return w_X;



    def Add_Derivative(     self,
                            D       : Derivative) -> None:
        """ Let w denote this weight function object. This method evaluates
        D(w) at w.Supported_Coords, and then stores the result in w.Derivatives.
        Critically, we only perform these calculations if w.Derivatives does not
        contain an entry corresponding to D.

        ------------------------------------------------------------------------
        Arguments:

        D : A derivative operator. In general, D(w) must make sense. This means
        that if w is a function on R^n, then D.Encoding should contain at most n
        elements. Why? D.Encoding[k] represents the number of derivatives with
        respect to the kth variable. This really only makes sense if w depends
        on at least k variables.

        ------------------------------------------------------------------------
        Returns:

        Nothing! """

        # First, check if we've already evaluated D(w).
        if(tuple(D.Encoding) in self.Derivatives):
            return;

        # If not, then let's calculate it.
        Dw : torch.Tensor = Evaluate_Derivative(    w       = self,
                                                    D       = D,
                                                    Coords   = self.Supported_Coords);
        self.Derivatives[tuple(D.Encoding)] = Dw.detach();

        # All done!
        return;



    def Get_Derivative(self, D : Derivative) -> torch.Tensor:
        """ 
        This function returns D(w) evaluated on the set of Coordinates we used 
        to initialize w. This assumes we already evaluated D(w) using the 
        Add_Derivative method.

        ------------------------------------------------------------------------
        Arguments:

        D : A derivative operator. We fetch the element of self.Derivatives
        corresponding to D.

        ------------------------------------------------------------------------
        Returns:

        The entry of self.Derivatives corresponding to D. This should be an
        1D array whose entries are D(w) at some set of coordinates. 
        """

        # First, get D(w) at self.Supported_Coords.
        Dw_Supported_Coords : torch.Tensor = self.Derivatives[tuple(D.Encoding)];

        # Next, extrapolate this derivative to the original set of Coordinates
        # we used to initialize w.
        Dw : torch.Tensor           = torch.zeros(self.Num_Coords, dtype = torch.float32);
        Dw[self.Supported_Indices]  = Dw_Supported_Coords;

        # All done!
        return Dw;



def Evaluate_Derivative(
        w           : Weight_Function,
        D           : Derivative,
        Coords      : torch.Tensor) -> torch.Tensor:
    """ 
    This function applies a particular partial derivative operator, D, of a 
    weight function, w, and then evaluates the resulting function on Coords. In 
    other words, this function evaluates D(w)(X), for each X in Coords.

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
    at the ith coordinate (ith row of Coords). 
    """

    # First, we need to convert Coords to double precision. We need this to
    # prevent nan's from appearing when computing high order derivatives.
    # we will convert back to single precision when we're done.
    Coords : torch.Tensor = Coords.to(dtype = torch.float64);

    # Next, we need to evaluate derivatives, so set Requires Grad to true.
    Coords.requires_grad_(True);

    # Make sure we can actually compute the derivatives. For this, we need
    # the input dimension of f to be <= the size of D's encoding vector.
    assert(D.Encoding.size <= w.Input_Dim);

    # We also need D.Encoding to be less (element-wise) than w.Powers.
    #Powers_np : numpy.ndarray = w.Powers.numpy();
    #assert(numpy.sum(D.Encoding >= Powers_np) == 0);

    # Now, let's get to work. The plan is the following: Suppose we want to find
    # D_t^{m(t)} D_x^{m(x)} D_y^{m(y)} D_z^{m(z)} w. First, we compute
    # D_t^{m(t)} w. From this, we calculate D_x^{m(x)} D_t^{m(t)} w, and so on.
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

    # Initialize Dx_Dt_w. If there are no x derivatives, then Dx_Dt_w = Dt_w.
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
        return Dx_Dt_w.to(dtype = torch.float32);

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
        return Dy_Dx_Dt_w.to(dtype = torch.float32);

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

    return Dz_Dy_Dx_Dt_w.to(dtype = torch.float32);
