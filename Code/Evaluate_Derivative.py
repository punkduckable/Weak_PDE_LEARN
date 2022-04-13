import  torch;

from Weight_Function    import Weight_Function;
from Derivative         import Derivative;



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
