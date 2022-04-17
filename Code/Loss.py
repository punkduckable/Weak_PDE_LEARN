import numpy;
import torch;
import math;

from typing import Tuple, List;

from Network            import Neural_Network;
from Integrate          import Integrate;
from Derivative         import Derivative;
from Library_Term       import Library_Term;
from Trial_Function     import Trial_Function;
from Weight_Function    import Weight_Function;


def Data_Loss(
        U           : Neural_Network,
        Inputs      : torch.Tensor,
        Targets     : torch.Tensor) -> torch.Tensor:
    """ This function evaluates the data loss, which is the mean square
    error between U at the Inputs, and the Targets. To do this, we
    first evaluate U at the data points. At each point (t, x), we then
    evaluate |U(t, x) - U'_{t, x}|^2, where U'_{t,x} is the data point
    corresponding to (t, x). We sum up these values and then divide by the
    number of data points.

    ----------------------------------------------------------------------------
    Arguments:

    U: The neural network which approximates the system response function.

    Inputs: If U is a function of one spatial variable, then this should
    be a two column tensor whose ith row holds the (t, x) coordinate of the
    ith data point. If U is a function of two spatial variables, then this
    should be a three column tensor whose ith row holds the (t, x, y)
    coordinates of the ith data point.

    Targets: If Targets has N rows, then this should be an N element
    tensor whose ith element holds the value of U at the ith data point.

    ----------------------------------------------------------------------------
    Returns:

    A scalar tensor whose sole entry holds the mean square data loss. """

    # Evaluate U at the data points.
    U_Predict = U(Inputs).view(-1);

    # Evaluate the pointwise square difference of U_Predict and Targets.
    Square_Error = ( U_Predict - Targets ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Weak_Form_Loss( U                   : Neural_Network,
                    Xi                  : torch.Tensor,
                    LHS_Term            : Library_Term,
                    RHS_Terms           : List[Library_Term],
                    Grid_Coords         : torch.Tensor,
                    V                   : float,
                    Weight_Functions    : List[Weight_Function]) -> torch.Tensor:
    """ To do :D """


    ############################################################################
    # First, determine the highest power of U that we need.

    Max_Pow         : int = LHS_Term.Trial_Function.Power;
    Num_RHS_Terms   : int = len(RHS_Terms);

    for i in range(Num_RHS_Terms):
        if(RHS_Terms[i].Trial_Function.Power > Max_Pow):
            Max_Pow = RHS_Terms[i].Trial_Function.Power;


    ############################################################################
    # Evaluate powers of U on the Grid

    # Now, evaluate U at the coords.
    U_Grid_Coords   : torch.Tensor  = U(Grid_Coords).view(-1);
    Num_Coords      : int           = U_Grid_Coords.numel();

    # Next, compute powers of U up to Max_Pow on Max_Grid. We will need these
    # values for when integrating.
    U_Coords_Powers : torch.Tensor  = torch.empty([Num_Coords, Max_Pow + 1], dtype = torch.float32);

    U_Coords_Powers[:, 0] = 1;

    for i in range(1, Max_Pow + 1):
        U_Coords_Powers[:, i] = torch.pow(U_Grid_Coords, i);


    ############################################################################
    # Construct the loss.
    # The loss takes the form ||A \xi - b||_2^2, where b \in R^M (M = Number of
    # weight functions) is defined by
    #       b_i = \int w_i(X) D(F(U(X))) dX
    # where w_i is the ith weight function and D(F(U)) is the RHS term.
    # Likewise, A \in R^{m x n} is defined by
    #       A_{i,j} = \int w_i(X) D_j(F_j(U(X))) dX
    # where D_j(F_j(U)) is the jth trial function.

    m : int             = len(Weight_Functions);
    n : int             = len(RHS_Terms);

    # Construct b.
    b : torch.Tensor    = torch.empty(m, dtype = torch.float32);

    for i in range(m):
        b[i] = Integrate(   w       = Weight_Functions[i],
                            D       = LHS_Term.Derivative,
                            FU_Grid = U_Coords_Powers[:, LHS_Term.Trial_Function.Power],
                            V       =  V);

    # Construct A \xi column by column
    A_Xi : torch.Tensor = torch.zeros(m, dtype = torch.float32);

    for j in range(n):
        D_j : Derivative        = RHS_Terms[j].Derivative;
        F_j : Trial_Function    = RHS_Terms[j].Trial_Function;
        A_j : torch.Tensor  = torch.empty(m, dtype = torch.float32);

        for i in range(m):
            A_j[i]  = Integrate(w       = Weight_Functions[i],
                                D       = D_j,
                                FU_Grid = U_Coords_Powers[:, F_j.Power],
                                V       = V);

        A_Xi += torch.multiply(A_j, Xi[j]);

    # Compute loss! (this is ||A \xi - b ||_2^2).
    return torch.sum((A_Xi - b)**2);



def Lp_Loss(Xi  : torch.Tensor,
            p   : float):
    """ This function approximates the L0 norm of Xi using the following
    quantity:
        w_1*|Xi[1]|^2 + w_2*|Xi[2]|^2 + ... + w_N*|Xi[N]|^2
    Where, for each k,
        w_k = 1/max{delta, |Xi[k]|^{p - 2}}.
    (where delta is some small number that ensures we're not dividing by zero!)

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    p: The "p" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        w_1*|Xi[1]|^p + w_2*|Xi[2]|^p + ... + w_N*|Xi[N]|^p
    where N is the number of components of Xi. """

    assert(p > 0 and p < 2)

    # First, square the components of Xi. Also, make a doule precision copy of
    # Xi that is detached from Xi's graph.
    delta : float = .000001;
    Xi_2          = torch.mul(Xi, Xi);
    Xi_Detach     = torch.detach(Xi);

    # Now, define a weights tensor.
    W               = torch.empty_like(Xi_Detach);
    N : int         = W.numel();
    for k in range(N):
        # First, obtain the absolute value of the kth component of Xi, as a float.
        Abs_Xi_k    : float = abs(Xi[k].item());

        # Now, evaluate W[k].
        W_k  = 1./max(delta, Abs_Xi_k**(2 - p));

        # Check for infinity (which can happen, unfortuneatly, if delta is too
        # small). If so, remedy it.
        if(math.isinf(W_k)):
            print("W_k got to infinty");
            print("Abs_Xi_k = %f" % Abs_Xi_k);
            W_k = 0;

        W[k] = W_k;

    # Finally, evaluate the element-wise product of Xi and W[k].
    W_Xi_2 = torch.mul(W, Xi_2);
    return W_Xi_2.sum();
