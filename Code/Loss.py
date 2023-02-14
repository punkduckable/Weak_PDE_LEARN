# Nonsense to add Classes directory to the Python search path.
import os
import sys

# Get path to Code, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Classes directory to the python path.
sys.path.append(Classes_Path);

import numpy;
import torch;
import math;

from typing import Tuple, List;

from Network            import Network, Rational;
from Integrate          import Integrate_PDE;
from Derivative         import Derivative;
from Library_Term       import Library_Term;
from Trial_Function     import Trial_Function;
from Weight_Function    import Weight_Function;



def Data_Loss(
        U           : Network,
        Inputs      : torch.Tensor,
        Targets     : torch.Tensor) -> torch.Tensor:
    """ 
    This function evaluates the data loss, which is the mean square
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

    A scalar tensor whose sole entry holds the mean square data loss. 
    """

    # Evaluate U at the data points.
    U_Predict = U(Inputs).view(-1);

    # Evaluate the point-wise square difference of U_Predict and Targets.
    Square_Error = ( U_Predict - Targets ) ** 2;

    # Return the mean square error.
    return Square_Error.mean();



def Weak_Form_Loss( U                   : Network,
                    Xi                  : torch.Tensor,
                    Mask                : torch.Tensor,
                    LHS_Term            : Library_Term,
                    RHS_Terms           : List[Library_Term],
                    Weight_Functions    : List[Weight_Function]) -> Tuple[torch.Tensor, torch.Tensor]:
    r""" 
    We assume the underlying PDE is
        D_0 F_0(U) = c_1 D_1 F_1(U) + ... + c_n D_K F_K(U).
    Since this equation is valid on the entire problem domain, Omega, we should
    also have
        \int_{\Omega} w D_0 F_0(U) dX = \sum_{k = 1}^{K} c_i \int_{\Omega} w D_k F_k(U) dX
    for any function, w, defined on Omega. Since the neural network, U, and
    vector Xi approximate the system response function and coefficient vector
    (c_1, ... , c_n), respectively, we expect that for each w,
        \int_{\Omega} w D_0 F_k(U) dX \approx \sum_{k = 1}^{K} Xi_i \int_{\Omega} w D_i F_i(U) dX

    This function calculates the left and right of the expression above for each
    function in Weight_Functions. This gives us a system of linear equations in
    the components of Xi. We report the mean square error of this system.
    In particular, this function calculates (1/m)||b - A*Xi||_2^2, where
    A \in R^{m x n} is the matrix whose i,j entry holds an approximation to the
    integral
            \int_{\Omega} w_i(X) D_j F_j(U(X)) dX.
    That is, A_{i,j} holds the value of the integral of the jth library term
    times the ith weight function. Likewise, b \in R^m is the vector whose
    ith entry holds the value
            \int_{\Omega} w_i(X) D_0 F_k(U(X)) dX.
    We approximate these integrals using a quadrature rule. We evaluate U at the
    Partition_Coords (which should be the same array of coordinates we used to
    initialize each weight function). We then use this to evaluate F_j(U) on the
    Partition_Coords. We then use this array of values, the weight function, and
    D_j to evaluate [A]_{i,j}.

    ----------------------------------------------------------------------------
    Arguments:

    U : The neural network which approximates the system response function.

    Xi : The 1D tensor of coefficients used to calculate the weak form loss 
    (see above). If there are n RHS Terms, then Xi should have n elements.

    Mask: A boolean tensor whose shape matches that of Xi. When adding the kth 
    RHS term to the Library_Xi product, we check if Mask[k] == False. If so, 
    We add 0*Xi[k]. Otherwise, we compute the integral of the kth library 
    term as usual.

    LHS_Term : The Left Hand Side library term. This is DF(U) in the PDE.

    RHS_Terms : The Right Hand Side library terms in the PDE - These are
    D_1 F_1(U), ... , D_n F_n(U) in the PDE.

    Weight_Functions : A list containing the weight functions we want to use
    to compute the Weak Form Loss.

    ----------------------------------------------------------------------------
    Returns:

    A two element tuple. The first element houses a single-element tensor whose 
    lone element holds the weak form loss. The second holds a 1D tensor whose 
    ith entry holds the difference between the ith component of b and the 
    ith component of A*xi. 
    """

    ############################################################################
    # Construct the loss.
    # The loss takes the form (1/m)||A \xi - b||_2^2, where b \in R^M (M = Number of
    # weight functions) is defined by
    #       b_i = \int w_i(X) D_0(F_0(U(X))) dX
    # where w_i is the ith weight function and D_0(F_0(U)) is the LHS term.
    # Likewise, A \in R^{m x n} is defined by
    #       A_{i,j} = \int w_i(X) D_j(F_j(U(X))) dX
    # where D_j(F_j(U)) is the jth RHS term.

    M : int             = len(Weight_Functions);
    K : int             = len(RHS_Terms);

    # Set up A, b.
    b       : torch.Tensor  = torch.empty(M, dtype = torch.float32);
    A_Xi    : torch.Tensor  = torch.zeros(M, dtype = torch.float32);

    for m in range(M):
        # First, compute the integrals for the kth weight function.
        wm_LHS, wm_RHSs = Integrate_PDE(w           = Weight_Functions[m], 
                                        U           = U,
                                        LHS_Term    = LHS_Term, 
                                        RHS_Terms   = RHS_Terms,
                                        Mask        = Mask);
        b[m]    = wm_LHS;
        for k in range(K):
            A_Xi += Xi[k]*wm_RHSs[k];
    
    # Compute loss! (this is (1/m)||A \xi - b ||_2^2).
    Residual : torch.Tensor = torch.subtract(b, A_Xi);
    return ((torch.sum(Residual**2))/float(m), Residual);



def Lp_Loss(Xi      : torch.Tensor, 
            Mask    : torch.Tensor,
            p       : float) -> torch.Tensor:
    """ 
    This function approximates the L0 norm of Xi using the following quantity:
        w_1*|Xi[1]|^2 + w_2*|Xi[2]|^2 + ... + w_N*|Xi[N]|^2
    Where, for each k,
        w_k =   1/max{delta, |Xi[k]|^{p - 2}}       if Mask[k] == False
                0                                   if Mask[k] == True
    (where delta is some small number that ensures we're not dividing by zero!)

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi vector in our setup. This should be a one-dimensional tensor.

    Mask: A boolean tensor whose shape matches that of Xi. When adding the kth 
    RHS term to the Library_Xi product, we check if Mask[k] == False. If so, 
    We add 0*Xi[k]. Otherwise, we compute the kth library term as usual.

    p: The "p" in in the expression above

    ----------------------------------------------------------------------------
    Returns:

        w_1*|Xi[1]|^p + w_2*|Xi[2]|^p + ... + w_N*|Xi[N]|^p
    where N is the number of components of Xi. 
    """

    assert(p > 0 and p < 2)

    # First, square the components of Xi. Also, make a double precision copy of
    # Xi that is detached from Xi's graph.
    delta : float = .0000001;
    Xi_2          = torch.mul(Xi, Xi);
    Xi_Detach     = torch.detach(Xi);

    # Now, define a weights tensor.
    W               = torch.empty_like(Xi_Detach);
    N : int         = W.numel();
    for k in range(N):
        if(Mask[k] == True):
            W[k] = 0.0;
            continue;

        # First, obtain the absolute value of the kth component of Xi, as a float.
        Abs_Xi_k    : float = abs(Xi[k].item());

        # Now, evaluate W[k].
        W_k  = 1./max(delta, Abs_Xi_k**(2 - p));

        # Check for infinity (which can happen, unfortunately, if delta is too
        # small). If so, remedy it.
        if(math.isinf(W_k)):
            print("W_k got to infinity");
            print("Abs_Xi_k = %f" % Abs_Xi_k);
            W_k = 0;

        W[k] = W_k;

    # Finally, evaluate the element-wise product of Xi and W[k].
    W_Xi_2 = torch.mul(W, Xi_2);
    return W_Xi_2.sum();



def L2_Squared_Loss(U : Network) -> torch.Tensor:
    """
    This function returns the sum of the squares of U's parameters. Suppose
    that U has P \in \mathbb{N} parameters (weights, biases, RNN coefficients). 
    Suppose we enumerate those parameters and let \Theta \in \mathbb{R}^P be 
    a vector whose ith element is the ith parameter in the enumeration. This 
    function returns ||U||_2^2.

    ---------------------------------------------------------------------------
    Arguments:

    U: A neural network object.

    ---------------------------------------------------------------------------
    Returns:

    A single element tensor whose lone element holds the square of the L2 norm 
    of U's parameter vector.
    """

    # Setup. 
    L2_Loss     : torch.Tensor  = torch.zeros(1, dtype = torch.float32);
    Num_Layers  : int           = U.Num_Layers;

    for i in range(Num_Layers):
        # We build up the loss one layer at a time. To do this, we add the 
        # L2 norm squared of the weight matrix and bias vector of each layer.
        W_i : torch.Tensor = U.Layers[i].weight;
        b_i : torch.Tensor = U.Layers[i].bias;

        Loss += torch.sum(torch.multiply(W_i, W_i));
        Loss += torch.sum(torch.multiply(b_i, b_i));
    
        # If this is a rational layer, we need to add its parameters.
        AF_i : torch.nn.Module = U.Activation_Functions[i];
        if(isinstance(AF_i, Rational)):
            Loss += torch.sum(torch.multiply(AF_i.a, AF_i.a));
            Loss += torch.sum(torch.multiply(AF_i.b, AF_i.b));

    # All done!
    return Loss;
