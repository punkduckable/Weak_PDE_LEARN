import torch;
import numpy;
import math;

from Derivative             import Derivative;



class Weight_Function(torch.nn.Module):
    """ 
    Objects of this class are bump functions with a specified center, X_0, and 
    radius, r. In particular, each object of this class represents the function 
    w : R^n -> R defined by
        w(X) =  { prod_{i = 1}^{n} exp(5 r^2/((X_i - X_0_i)^2 - r^2) + 5)   if ||X - X_0||_{max} < r
                { 0                                                         otherwise
    where p_1, ... , p_n are positive integers. Such a function is C^infinity

    A weight function is always paired with a set of coordinates. We assume 
    these coordinates lie on a regular grid. The associated grid lines 
    partition the problem domain into a set of rectangles. We assume each 
    rectangle has a volume of V. We use these coordinates to compute various 
    integrals of the form
            \int_{S} w(X) D^{(k)} F_k(U(X)) dX),
    for some set S. Here, D^{(k)} is a partial derivative operator, and F_k(U)
    is a trial function. To approximate this integral, we use the n-dimensional 
    composite trapezoidal rule on the grid. It can be shown that in this case 
    (and assuming the integrand is zero along the boundary), the trapezoidal 
    rule in this case is equal to the sum over i of the integrand evaluated at
    the ith grid point, multiplied by V (volume of any onf of the rectangles 
    engendered by the regular grid).
    
    Using the same set of coordinates for each integral also allows for a key
    performance breakthrough: We only need to evaluate w and its derivatives at
    the coordinates once. Once we have these values, we can use them each time 
    we want to compute an integral involving w (or its derivatives). This is 
    why weight function objects have a "Derivatives" dictionary. This 
    dictionary stores the various derivatives of w, evaluated at the set of 
    coordinates you used to initialize w. 

    We can, however, make a critical reduction that boosts performance and
    slashes storage requirements: A bump function and its derivatives are zero
    outside of the ball B_r(X_0) (using the infinity norm). As such, there is 
    no need to store the value of w or its derivatives at points outside of
    B_r(X_0). Weight functions take advantage of this. In particular, when you
    initialize a weight function, you do so using that function's set of
    coordinates. We determine which coordinates are in B_r(X_0) and down write
    them (and their indices within the sef of coordinates). When the user asks
    us to calculate a particular derivative of w, we simply evaluate it at the
    coordinates we wrote down. If the user asks for the value of a derivative 
    of w, we can use the values we wrote down to reconstruct the derivative on 
    the entire set of coordinates.

    ---------------------------------------------------------------------------
    Members:

    X_0 : The center of the bump function, as defined above. This should be a
    1D tensor whose length is n (the dimension of the domain of w).

    r : the radius of the bump function. The bump function is zero outside of
    B_r(X_0).

    Input_Dim : w (see above) is defined on R^n. This is n.

    Supported_Coords : When we initialize a weight function object, we pass a
    set of coordinates. We partition the problem domain using a set of grid 
    parallel to each axis. The coordinates we pass is the resulting set of 
    grid-points that this partition engenders. Supported_Coords is the set of 
    coordinates in that set that are also in B_r(X_0).

    V : The volume of any sub-rectangle in the the partition of the problem 
    domain.

    Derivatives : A dictionary that holds various derivatives of w evaluated at
    Supported_Coords. For this dictionary, an elements key is a Derivative
    object's encoding array. Its value is a torch tensor that holds the value 
    of the corresponding derivative of the weight function at a set of 
    coordinates. 
    """

    def __init__(   self,
                    X_0     : torch.Tensor,
                    r       : float,
                    Coords  : torch.Tensor, 
                    V       : float) -> None:
        """ 
        Class initializer.

        -----------------------------------------------------------------------
        Arguments:

        X_0 : The center of the weight function. See class doc-string.

        r : The radius of the weight function. See class doc-string.

        Coords : A list of coordinates. Whenever we apply a derivative to w, we
        evaluate (and store) the derivative at the coordinates in Coords. We
        assume this is a B by n dimensional array whose ith row holds the ith
        coordinate. Here, B is the number of coordinates and n is the number of
        components in each coordinate. 

        V : The volume of any sub-rectangle in the the partition of the problem 
        domain.
        """

        # First, call the module initializer.
        super(Weight_Function, self).__init__();


        ########################################################################
        # Checks

        assert(len(X_0.shape)       == 1);              # X_0 should be a 1D Tensor.
        assert(len(Coords.shape)    == 2);              # Coords should be a 2D Tensor.
        assert(Coords.shape[1]      == X_0.shape[0]);   # X_0 should be in the same space as the Coords
        assert(r > 0);                                  # radius must be positive.


        ########################################################################
        # Assuming we passed the checks, assign X_0 and r members. We can also
        # determine the input dimension from the length of X_0.

        self.X_0            : torch.Tensor  = X_0;
        self.Input_Dim      : int           = X_0.numel();
        self.r              : float         = r;
        self.V              : float         = V;

        # Get Num_Coords.
        self.Num_Coords     : int           = Coords.shape[0];

        # Initialize this object's derivatives array.
        self.Derivatives    : dict = {};


        ########################################################################
        # Now, determine which coordinates are in B_r(X_0).

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0                        : torch.Tensor  = torch.subtract(Coords, X_0);
        Max_XmX0                    : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);

        # Now, determine which coordinates are in B_r(X_0).
        Supported_Indices           : torch.Tensor  = torch.less(Max_XmX0, self.r);

        # Record the coordinates that are.
        self.Supported_Coords       : torch.Tensor  = Coords[Supported_Indices, :];



    # Product of 1D bumps. If using, make init use inf norm to determine
    # supported points.
    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """ 
        This method evaluates the weight function at each coordinate of X.

        -----------------------------------------------------------------------
        Arguments:

        X : A 2D tensor, each row of which holds a coordinate at which we want
        to evaluate the weight function. If this weight function object is
        defined on R^n, then X should be a B by n tensor.

        -----------------------------------------------------------------------
        Returns:

        A B element tensor whose ith entry is w evaluated at the ith coordinate
        (ith row of X). 
        """

        # w(X) =    { prod_{i = 1}^{n} exp(5*r^2/((X_i - X_0_i)^2 - r^2) + 5)   if ||X - X_0||_{max} < r
        #           { 0                                                         otherwise

        # First, calculate || X - X_0 ||_{infinity}.
        XmX0                    : torch.Tensor  = torch.subtract(X, self.X_0);
        Max_XmX0                : torch.Tensor  = torch.linalg.vector_norm(XmX0, ord = float('inf'), dim = 1);

        # Determine which coordinates are in B_r(X_0).
        Supported_Indices  : torch.Tensor  = torch.less(Max_XmX0, self.r);

        # Evaluate (X_i - X_0_i)^2 - r^2 for each component of each
        # supported coordinate.
        XmX0                    : torch.Tensor  = XmX0[Supported_Indices, :];
        Denominators            : torch.Tensor  = torch.subtract(torch.pow(XmX0, 2.), (self.r)**2);

        # Compute w at the supported coordinates.
        r2_XmX02_r2_5           : torch.Tensor = torch.add(torch.divide(torch.full_like(Denominators, 7.5*(self.r)**2), Denominators), 7.5);
        Exponent                : torch.Tensor = torch.sum(r2_XmX02_r2_5, dim = 1);
        w_X_Supported           : torch.Tensor = torch.exp(Exponent);

        # Calculate w at the rest of the coordinates.
        w_X                     : torch.Tensor  = torch.zeros(X.shape[0], dtype = X.dtype);
        w_X[Supported_Indices]                  = w_X_Supported;

        # Return!
        return w_X;



    def Add_Derivative(     self,
                            D       : Derivative) -> None:
        """ 
        Let w denote this weight function object. This method evaluates D(w) at 
        w.Supported_Coords, and then stores the result in w.Derivatives. 
        Critically, we only perform these calculations if w.Derivatives does 
        not contain an entry corresponding to D.

        -----------------------------------------------------------------------
        Arguments:

        D : A derivative operator. In general, D(w) must make sense. This means
        that if w is a function on R^n, then D.Encoding should contain at most 
        n elements. Why? D.Encoding[k] represents the number of derivatives 
        with respect to the kth variable. This really only makes sense if w 
        depends on at least k variables.

        -----------------------------------------------------------------------
        Returns:

        Nothing! 
        """

        # First, check if we've already evaluated D(w).
        if(tuple(D.Encoding) in self.Derivatives):
            return;

        # If not, then let's calculate it.
        Dw : torch.Tensor = Evaluate_Derivative(    w       = self,
                                                    D       = D,
                                                    Coords  = self.Supported_Coords);
        self.Derivatives[tuple(D.Encoding)] = Dw.detach();

        # All done!
        return;



    def Get_Derivative(self, D : Derivative) -> torch.Tensor:
        """ 
        This function returns D(w) evaluated on the set of Coordinates we used 
        to initialize w. This assumes we already evaluated D(w) using the 
        Add_Derivative method.

        -----------------------------------------------------------------------
        Arguments:

        D : A derivative operator. We fetch the element of self.Derivatives
        corresponding to D.

        -----------------------------------------------------------------------
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



def Build_From_Other( 
        X_1     : torch.Tensor, 
        r_1     : float,
        W_0     : Weight_Function) -> Weight_Function:
    """
    This function builds a new weight function from an existing one. Why is 
    this possible? Recall that a weight function, w, is defined as follows
        w_0(X) = prod_{i = 1}^{N} exp(c/(((X_i - X_0_i)/r_0)^2 - 1) + c)
    (if ||X - X_0||_2 < r_0, and 0 otherwise). Suppose that we have a second 
    weight function, w_1, which has the same radius but a different center, 
    X_1, and radius, r_1. Then, notice that
        w_1((X - X_0)r_1/r_0 + X_1) = prod_{i = 1}^{N} exp(c/(((X_i - X_0_i)(r_1/r_0)/r_1)^2 - 1) + c)
                                    = prod_{i = 1}^{M} exp(c/(((X_i - X_0_i)/r_0)^2 - 1) + c)
                                    = w_0(X)
    Likewise,
        w_0((X - X_1)r_0/r_1 + X_0) = prod_{i = 1}^{N} exp(c/(((X_i - X_1_i)r_0/(r_0*r_1))^2 - 1) + c)
                                    = prod_{i = 1}^{N} exp(c/(((X_i - X_1_i)/r_1)^2 - 1) + c)
                                    = w_1(X)
    In particular, this means that
        (d/dx_i)w_1(X) = (r_0/r_1)(d/dx_i)w_0((X - X_1)r_0/r_1 + X_0)
    Thus, we can easily relate the partial derivatives of w_0 to w_1. In 
    particular, if we already know the derivatives of w_0, and we can 
    obtain the gird points of w_1 by applying the mapping 
        X -> (X - X_1)r_0/r_1 + X_0
    to the grid points of w_0, then then don't need to compute the derivatives
    of w_1 from scratch... instead, we can simply fetch those of w_1. This is 
    precisely the approach that we take here. 

    In particular, we build a new weight function, W_1, from an existing 
    weight function, W_0. To do this, we simply apply the mapping 
        X -> (X - X_1)r_0/r_1 + X_0
    to the grid points in W_0. We then copy the derivatives of W_0 to 
    those in W_1 by multiplying the ith derivative of W_0 by (r_0/r_1)^k(i), 
    where k(i) is the order of the ith derivative.
    """

    # First, fetch the relevant parameters from W_0.
    X_0         : torch.Tensor  = W_0.X_0;
    r_0         : float         = W_0.r;
    W_0_Coords  : torch.Tensor  = W_0.Supported_Coords;
    V_0         : float         = W_0.V;

    # Fetch the number of dimensions. We need this to adjust V.
    n           : int           = W_0_Coords.shape[1];
    
    # Build new variables. 
    W_1_Coords  : torch.Tensor  = (W_0_Coords - X_0)*(r_1/r_0) + X_1;
    V_1         : float         = V_0*((r_1/r_0)**n);

    # Initialize W_1
    W_1 = Weight_Function(X_0 = X_1, r = r_1, Coords = W_1_Coords, V = V_1);

    # Now, build the derivatives of W_1 from those of W_0. 
    W_0_Derivatives_Dict = W_0.Derivatives;
    for Encoding, W_0_Derivative in W_0_Derivatives_Dict.items():
        # First, determine the order of this derivative.
        Order : int = 0;
        for i in range(len(Encoding)):
            Order += Encoding[i];
        
        # Next, scale W_0_Derivative by (r_0/r_1)^Order to get the  
        # corresponding derivatives for W_1. This works because of how we 
        # define the coordinates of W_1. 
        W_1_Derivative : torch.Tensor = W_0_Derivative*((r_0/r_1)**Order);
        W_1.Derivatives[Encoding] = W_1_Derivative;
    
    # All done!
    return W_1;



def Evaluate_Derivative(
        w           : Weight_Function,
        D           : Derivative,
        Coords      : torch.Tensor) -> torch.Tensor:
    """ 
    This function applies a particular partial derivative operator, D, of a 
    weight function, w, and then evaluates the resulting function on Coords. In 
    other words, this function evaluates D(w)(X), for each X in Coords.

    ---------------------------------------------------------------------------
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
    which we want to evaluate D(w). Thus, each row of D(w) represents a point 
    in R^n.

    ---------------------------------------------------------------------------
    Returns:

    This returns a 1D Tensor. If Coords has B rows, then the returned tensor 
    has B elements. The ith component of the returned Tensor holds D(w) 
    evaluated at the ith coordinate (ith row of Coords). 
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
