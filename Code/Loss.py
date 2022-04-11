import numpy;
import torch;
import math;

from Network  import    Neural_Network;
from Evaluate_Derivatives import Evaluate_Derivatives;



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
    U_Predict = U(Inputs).squeeze();

    # Evaluate the pointwise square difference of U_Predict and Targets.
    Square_Error = ( U_Predict - Targets ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Weak_Form_Loss(None) -> torch.Tensor:
    """ To do.... """

    return torch.tensor(0, dtype = numpy.float32);



def Lp_Loss(Xi : torch.Tensor, p : float):
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
