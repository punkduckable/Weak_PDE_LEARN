# Nonsense to add Classes diectory to the Python search path.
import os
import sys

# Get path to Code, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Classes directory to the python path.
sys.path.append(Classes_Path);

import  numpy       as np;
import  torch;
from    typing      import List, Tuple;

from Network            import Neural_Network;
from Loss               import Data_Loss, Weak_Form_Loss, Lp_Loss;
from Library_Term       import Library_Term;
from Weight_Function    import Weight_Function;



def Training(
        U                                   : Neural_Network,
        Xi                                  : torch.Tensor,
        Inputs                              : torch.Tensor,
        Targets                             : torch.Tensor,
        LHS_Term                            : Library_Term,
        RHS_Terms                           : List[Library_Term],
        Partition                           : torch.Tensor,
        V                                   : float,
        Weight_Functions                    : List[Weight_Function],
        p                                   : float,
        Lambda                              : float,
        Optimizer                           : torch.optim.Optimizer,
        Device                              : torch.device = torch.device('cpu')) -> None:
    """ This function runs one epoch of training. We enforce the learned PDE
    (library-Xi product) at the Coll_Points. We also make U match the
    Targets at the Inputs.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coeffcients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If U accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Targets: A tensor holding the value of the true solution at the data
    points. If Inputs has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    LHS_Term: The Library Term (trial functiom + derivative) that appears on
    the left hand side of the PDE. This is generally a time derivative of U.

    RHS_Terms: A list of the Library terms (trial function + derivative)
    that appear in the right hand side of the PDE.

    Partition: A 2D array whose ith row holds the coordinates of the ith
    point in the partition of the problem domain. We use these points as
    quadrature points when approximating the integrals in the weak form loss.
    We also assume each weight function was initialized with Coords = Partition.

    V: The volume of any sub-rectangle induced by the partition. We
    assume that along any dimension, the partition points are uniformly spaced,
    meaning that every sub-rectangle induced by the partition has the same
    volume. V is that volume.

    Weight_Functions: A list of the weight functions in the weak form loss.

    Index_to_Derivatives: A mapping which sends sub-index values to spatial
    partial derivatives. This is needed to build the library in Coll_Loss.
    If U is a function of 1 spatial variable, this should be the function
    Index_to_x_Derivatives. If U is a function of two spatial variables, this
    should be an instance of Index_to_xy_Derivatives.

    Col_Number_to_Multi_Index: A mapping which sends column numbers to
    Multi-Indices. Coll_Loss needs this function. This should be an instance of
    the Col_Number_to_Multi_Index_Class class.

    p, Lambda: the settings value for p and Lambda (in the loss function).

    optimizer: the optimizer we use to train U and Xi. It should have
    been initialized with both network's parameters.

    Device: The device for U and Xi.

    ----------------------------------------------------------------------------
    Returns:

    Nothing! """

    # Put U in training mode.
    U.train();

    # Define closure function (needed for LBFGS)
    def Closure():
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Evaluate the Loss
        Loss = (Weak_Form_Loss( U                   = U,
                                Xi                  = Xi,
                                LHS_Term            = LHS_Term,
                                RHS_Terms           = RHS_Terms,
                                Partition           = Partition,
                                V                   = V,
                                Weight_Functions    = Weight_Functions)

                +

                Data_Loss(
                    U                   = U,
                    Inputs              = Inputs,
                    Targets             = Targets)

                +

                Lambda*Lp_Loss( Xi      = Xi,
                                p       = p));

        # Back-propigate to compute gradients of Loss with respect to network
        # parameters (only do if this if the loss requires grad)
        if (Loss.requires_grad == True):
            Loss.backward();

        return Loss;

    # update network parameters.
    Optimizer.step(Closure);



def Testing(
        U                                   : Neural_Network,
        Xi                                  : Neural_Network,
        Inputs                              : torch.Tensor,
        Targets                             : torch.Tensor,
        LHS_Term                            : Library_Term,
        RHS_Terms                           : List[Library_Term],
        Partition                           : torch.Tensor,
        V                                   : float,
        Weight_Functions                    : List[Weight_Function],
        p                                   : float,
        Lambda                              : float,
        Device                              : torch.device = torch.device('cpu')) -> Tuple[float, float]:
    """ This function evaluates the losses.

    Note: You CAN NOT run this function with no_grad set True. Why? Because we
    need to evaluate derivatives of U with respect to its inputs to evaluate
    Coll_Loss! Thus, we need torch to build a computational graph.

    ----------------------------------------------------------------------------
    Arguments:

    U: The network that approximates the PDE solution.

    Xi: The vector that stores the coeffcients of the library terms.

    Coll_Points: the collocation points at which we enforce the learned
    PDE. If U accepts d spatial coordinates, then this should be a d+1 column
    tensor whose ith row holds the t, x_1,... x_d coordinates of the ith
    Collocation point.

    Inputs: A tensor holding the coordinates of the points at which we
    compare the approximate solution to the true one. If u accepts d spatial
    coordinates, then this should be a d+1 column tensor whose ith row holds the
    t, x_1,... x_d coordinates of the ith Datapoint.

    Targets: A tensor holding the value of the true solution at the data
    points. If Targets has N rows, then this should be an N element tensor
    of floats whose ith element holds the value of the true solution at the ith
    data point.

    Time_Derivative_Order: We try to solve a PDE of the form (d^n U/dt^n) =
    N(U, D_{x}U, ...). This is the 'n' on the left-hand side of that PDE.

    LHS_Term: The Library Term (trial functiom + derivative) that appears on
    the left hand side of the PDE. This is generally a time derivative of U.

    RHS_Terms: A list of the Library terms (trial function + derivative)
    that appear in the right hand side of the PDE.

    Partition: A 2D array whose ith row holds the coordinates of the ith
    point in the partition of the problem domain. We use these points as
    quadrature points when approximating the integrals in the weak form loss.
    We also assume each weight function was initialized with Coords = Partition.

    V: The volume of any sub-rectangle induced by the partition. We
    assume that along any dimension, the partition points are uniformly spaced,
    meaning that every sub-rectangle induced by the partition has the same
    volume. V is that volume.

    Weight_Functions: A list of the weight functions in the weak form loss.

    p, Lambda: the settings value for p and Lambda (in the loss function).

    Device: The device for Sol_NN and PDE_NN.

    ----------------------------------------------------------------------------
    Returns:

    a tuple of floats. The first element holds the Data_Loss. The second
    holds the Weak_Form_Loss. The third holds Lambda times the Lp_Loss. """

    # Put U in evaluation mode
    U.eval();

    # Get the losses
    Data_Loss_Value : float  = Data_Loss(
                                U           = U,
                                Inputs      = Inputs,
                                Targets     = Targets).item();

    Weak_Form_Loss_Value : float = Weak_Form_Loss(
                                        U                   = U,
                                        Xi                  = Xi,
                                        LHS_Term            = LHS_Term,
                                        RHS_Terms           = RHS_Terms,
                                        Partition           = Partition,
                                        V                   = V,
                                        Weight_Functions    = Weight_Functions).item()

    Lambda_Lp_Loss_Value : float = Lambda*Lp_Loss(
                                            Xi    = Xi,
                                            p     = p).item();

    # Return the losses.
    return (Data_Loss_Value, Weak_Form_Loss_Value, Lambda_Lp_Loss_Value);
