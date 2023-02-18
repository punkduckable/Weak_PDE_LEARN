# Nonsense to add Code, Classes directories to the Python search path.
import os
import sys

# Get path to parent directory
Main_Path       = os.path.dirname(os.path.abspath(os.path.curdir));

# Get the path to the Code, Classes directories.
Code_Path       = os.path.join(Main_Path, "Code");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add them to the Python search path.
sys.path.append(Main_Path);
sys.path.append(Code_Path);
sys.path.append(Classes_Path);

# Code files.
from    Network                     import  Network;
from    Derivative                  import  Derivative;
from    Library_Term                import  Library_Term;
from    Derivative_From_Derivative  import  Derivative_From_Derivative;

from    typing                      import  Dict, List, Tuple;
import  torch;
import  numpy;



def PDE_Residual(
        U           : Network,
        Xi          : torch.Tensor,
        Mask        : torch.Tensor,
        Coll_Points : torch.Tensor,
        LHS_Term    : Library_Term,
        RHS_Terms   : List[Library_Term],
        Device      : torch.device = torch.device('cpu')) -> torch.Tensor:
    r""" 
    This is an adaptation of the Collocation Loss function from PDE-LEARN. 
    Since Weak PDE-LEARN does not enforce the PDE directly in its loss 
    function, the base algorithm has no collocation points. However, 
    visualizing the PDE residual on the problem domain can be very helpful, 
    which is why this function exists. 

    Suppose the underlying PDE is
        D_0 F_0 U = \sum_{k = 1}^{K} c_k (D_k F_k U).
    We can then define the "PDE Residual" as follows:
        PDE_Residual(t, x) = D_0 F_0 U(t, x) - \sum_{k = 1}^{K} c_k (D_k F_k U(t, x)).
    This function evaluates the PDE residual at each collocation point, and 
    then returns this information in a tensor.

    ---------------------------------------------------------------------------
    Arguments:

    U: The Neural Network that approximates the solution.

    Xi: A trainable (requires_grad = True) torch 1D tensor. If there are N
    RHS_Terms, this should be an N element vector.

    Mask: A boolean tensor whose shape matches that of Xi. When adding the kth 
    RHS term to the Library_Xi product, we check if Mask[k] == False. If so, 
    We add 0*Xi[k]. Otherwise, we compute the kth library term as usual.

    Coll_Points: B by n column tensor, where B is the number of coordinates and
    n is the dimension of the problem domain. The ith row of Coll_Points should
    hold the components of the ith collocation points.

    LHS_Term : A Term object representing T_0 in the equation above.

    RHS_Terms : A list of Term objects whose ith entry represents T_i in the
    equation above.

    Device: The device (gpu or cpu) that we train on.

    ---------------------------------------------------------------------------
    Returns:

    A 1D tensor whose ith entry holds the PDE residual at the ith collocation
    point. 
    """

    # Make sure Xi's length matches RHS_Terms'.
    assert(torch.numel(Xi) == len(RHS_Terms));

    # Make sure Coll_Points requires grad.
    Coll_Points.requires_grad_(True);



    ###########################################################################
    # Evaluate U and its powers at the collocation points

    # First, evaluate U at the Coll_Points.
    U_Coords : torch.Tensor = U(Coll_Points).view(-1);

    # Second, determine the highest power of U that we need.
    Max_Pow         : int = LHS_Term.Trial_Function.Power;
    Num_RHS_Terms   : int = len(RHS_Terms);

    for i in range(Num_RHS_Terms):
        if(RHS_Terms[i].Trial_Function.Power > Max_Pow and Mask[i] == False):
            Max_Pow = RHS_Terms[i].Trial_Function.Power;

    # Next, compute powers of U up to Max_Pow on the partition. We will need
    # these values for when integrating. We store them in a list.
    U_Coords_Powers = [];

    U_Coords_Powers.append(torch.ones(Coll_Points.numel(), dtype = torch.float32));
    U_Coords_Powers.append(U_Coords);
    for i in range(2, Max_Pow + 1):
        U_Coords_Powers.append(torch.pow(U_Coords, i));



    ###########################################################################
    # Evaluate the LHS term on the collocation coordinates. 
    
    # First, set up an "identity" derivative.
    I : Derivative = Derivative(Encoding = numpy.array([0, 0]));

    # Compute f_0 U = D_0 F_0 U. 
    PDE_LHS : torch.Tensor = Derivative_From_Derivative(
                            Da      = LHS_Term.Derivative,
                            Db      = I,
                            Db_U    = U_Coords_Powers[LHS_Term.Trial_Function.Power],
                            Coords  = Coll_Points).view(-1);



    ###########################################################################
    # Now compute the RHS terms on the collocation coordinates.

    PDE_RHS_Terms_List : List[torch.Tensor] = [];
    
    for i in range(Num_RHS_Terms):
        if(Mask[i] == False):
            PDE_RHS_Terms_List.append(Derivative_From_Derivative(
                                Da      = RHS_Terms[i].Derivative,
                                Db      = I,
                                Db_U    = U_Coords_Powers[RHS_Terms[i].Trial_Function.Power],
                                Coords  = Coll_Points).view(-1));
        else:
            PDE_RHS_Terms_List.append(torch.zeros_like(PDE_LHS));

    PDE_RHS : torch.Tensor = torch.zeros_like(U_Coords);
    for i in range(Num_RHS_Terms):
        PDE_RHS += Xi[i]*PDE_RHS_Terms_List[i];



    ###########################################################################
    # Finally, evaluate the PDE residual at the collocation coordinates. 

    return torch.subtract(PDE_LHS, PDE_RHS);
