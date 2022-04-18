import torch;

from Weight_Function    import Weight_Function;
from Derivative         import Derivative;



def Integrate(w             : Weight_Function,
              D             : Derivative,
              FU_Partition  : torch.Tensor,
              V             : float) -> torch.Tensor:
    """ This function approximates the integral
            \int_{\Omega} w(X) D F(U(X)) dX
    where \Omega is the integration domain, w is a weight function, D is a
    partial derivative operator, and F(U) is a trial function. We assume both
    w and U are functions from R^n to R, for some n. We first use integration
    by parts to rewrite this integral. For this, we assume that w and all of
    its derivatives are zero on the boundary of \Omega. With this assumption,
    the integral above is equal to
            (-1)^{|D|} \int_{\Omega} D w(X) F(U(X)) dX
    where |D| is the sum of the partial derivatives in D. For example, if
    D = D_x^2 D_y^3 D_z^5, then |D| = 2 + 3 + 5 = 10. We approximate the
    integral using the composite trapezodial rule for R^n. For this, we assume
    that the domain, \Omega, is a rectangle. That is,
        \Omega = [a_1, b_1] x ... x [a_n, b_n]
    for some a_1, ... , a_n and b_1, ... , b_n. We assume the user has
    partitioned \Omega using a uniform partition (for each dimension, the
    partition points along that dimension are evenly spaced. Girdlines along
    different dimensions, however, may have different spacing between them).
    Under this assumption, the partition partitions Omega into a set of smaller
    rectangles. Since the gird is uniform, each sub-rectangle has the same
    volume. Let V denote that volume. Further, let {X_1, ... , X_N} denote the
    set points in the partition. Since w is, by assumption, zero along the
    boundary of Omega, it must be zero at each point on the boundary of Omega.
    With this assumption, it can be shown that the composite trapezodial rule
    apprpxoimation to the integral is given by
            (-1)^{|D|} V*( \sum_{i = 1}^{N} D w(X_i) F(U(X_i)) )
    (in the trapezodial rule in R^n, gird points along the boundary are weighted
    differently from those inside the boundary. However, the integrand is zero
    along the boundary points, we can safely ignore those differences and
    pretend that all points are weighted evenly; the points in the summation
    above with the wrong weighting evaluate to zero anyway, so the incorrect
    weighting is moot). The quantity above is precisely what this function
    evaluates.

    ----------------------------------------------------------------------------
    Arguments:

    w : This is a weight function. We assume that the user has already
    calculated D(w) (via the "Add_Derivative" method) on the partition.

    D : This is a derivative operator.

    FU_Partition : We assume this is a 1D tensor whose ith entry holds the value
    of a trial function, F(U), at the ith partition point. That is,
        FU_Partition[i] = F(U(X_i)).
    We assume that the user evaluated FU_Partition on the same partition used to
    evaluate D(w). Thus, the ith entry of w.Derivatives[D] is D(w)(X_i).

    V : This is the volume of each sub-rectangle in the partition of Omega (see
    above).

    ----------------------------------------------------------------------------
    Returns:

    A single element tensor whose lone entry holds the value
        (-1)^{|D|} V*( \sum_{i = 1}^{N} D w(X_i) F(U(X_i))) ) """

    # First, determine the indicies of the partition points (subset of
    # {1, 2, ... , N}) in the support of w.
    Supported_Indices : torch.Tensor = w.Supported_Indices;

    # Next, extract the corresponding elements of FU_Partition.
    FU_Partition_Supported : torch.Tensor = FU_Partition[Supported_Indices];

    # Get D(w) evaluated on the partition points in the support of w.
    Dw_Partition_Supported : torch.Tensor = w.Derivatives[tuple(D.Encoding)];

    # Sum the element-wise product of FU_Partition_Supported and D(w). This yields
    # the summation \sum_{i = 1}^{N} D w(X_i) F(U(X_i)).
    Sum : torch.Tensor = torch.sum(torch.multiply(FU_Partition_Supported, Dw_Partition_Supported));

    # Finally, multiply Sum by (-1)^{|D|}V, yielding the integral approximation
    return torch.multiply(Sum, V*((-1)**D.Order));
