# Nonsense to add Classes directory to the Python search path.
import os
import sys

# Get path to Code, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Classes directory to the python path.
sys.path.append(Classes_Path);

import  numpy       as np;
import  torch;
from    typing      import List, Tuple, Dict;

from Network            import Network;
from Loss               import Data_Loss, Weak_Form_Loss, Lp_Loss, L2_Squared_Loss;
from Library_Term       import Library_Term;
from Weight_Function    import Weight_Function;



def Training(
        U_List                              : List[Network],
        Xi                                  : torch.Tensor,
        Mask                                : torch.Tensor, 
        Inputs_List                         : List[torch.Tensor],
        Targets_List                        : List[torch.Tensor],
        LHS_Term                            : Library_Term,
        RHS_Terms                           : List[Library_Term],
        Weight_Functions_List               : List[List[Weight_Function]],
        p                                   : float,
        Weights                             : Dict[str, float],
        Optimizer                           : torch.optim.Optimizer,
        Device                              : torch.device = torch.device('cpu')) -> Dict:
    """ 
    This function runs one epoch of training. For each network, we evaluate 
    the weak form , data, and L2 losses.. We also evaluate the Lp loss of 
    Xi. Once we evaluate each loss, we compute the total loss and then use 
    back-propagation to update each network's parameters (along with xi).

    ---------------------------------------------------------------------------
    Arguments:

    U_List: A list of Networks whose ith element holds the network that 
    approximates the ith system response function.

    Xi: The vector that stores the coefficients of the library terms.

    Mask: A boolean tensor whose shape matches that of Xi. We use this to 
    compute the Collocation loss (see that doc string).

    Inputs_List: A list of tensors whose ith element holds the coordinates of 
    the points at which we evaluate the U_List[i]. If the ith system response 
    function is a function of d spatial coordinates, then the ith list item 
    should be a d+1 column tensor whose kth row holds the t, x_1,... x_d 
    coordinates of the kth Data-point for the ith system response function.

    Targets_List: A list of Tensors whose ith entry holds the value of the ith 
    system response function at Inputs_List[i]. If Inputs_List[i] has N rows, 
    then this should be an N element tensor of floats whose kth element holds 
    the value of the ith true solution at the kth row of Inputs_List[k].

    LHS_Term: The Library Term (trial function + derivative) that appears on
    the left hand side of the PDE. This is generally a time derivative of U.

    RHS_Terms: A list of the Library terms (trial function + derivative)
    that appear in the right hand side of the PDE.

    Weight_Functions_List: A list of lists. The ith list item holds the list 
    of weight functions that we use to enforce the weak form loss for the 
    ith system response function. 

    p: the settings value for p in "Lp" loss function.

    Weights: A dictionary of floats. It should have keys for "Lp", "Weak", and 
    "Data".

    Optimizer: the optimizer we use to train U and Xi. It should have been 
    initialized with both network's parameters.

    Device: The device for U and Xi.

    ----------------------------------------------------------------------------
    Returns:

    A dictionary with the following keys:
        "Weak Form Loss", "Data Loss", "L2 Loss": lists of floats whose ith 
        entry holds the corresponding loss for the ith data set / system 
        response function.

        "Total Loss": a list of floats whose ith entry houses the total loss 
        for the ith data set.

        "Lp Loss": A float housing the value of the Lp loss.
    """

    assert(len(U_List) == len(Weight_Functions_List));
    assert(len(U_List) == len(Inputs_List));
    assert(len(U_List) == len(Targets_List));

    Num_DataSets : int = len(U_List);

    # Put each U in training mode.
    for i in range(Num_DataSets):
        U_List[i].train();

    # Initialize variables to track the residual, losses. We need to do this
    # because we find these variables in the Closure function (which has its own
    # scope. Thus, any variables created in Closure are inaccessible from
    # outside Closure).
    Residual_List       : List[float] = [];
    Weak_Loss_List      : List[float] = [0]*Num_DataSets;
    Data_Loss_List      : List[float] = [0]*Num_DataSets;
    L2_Loss_List        : List[float] = [0]*Num_DataSets;
    Lp_Loss_Buffer      = 0.0;
    Total_Loss_List     : List[float] = [0]*Num_DataSets;

    for i in range(Num_DataSets):
        Residual_List.append(torch.empty(len(Weight_Functions_List[i]), dtype = torch.float32));

    # Define closure function (needed for LBFGS)
    def Closure():
        # Zero out the gradients (if they are enabled).
        if (torch.is_grad_enabled()):
            Optimizer.zero_grad();

        # Set up buffers to hold the losses
        Weak_Loss_Value     = torch.zeros(1, dtype = torch.float32);
        Data_Loss_Value     = torch.zeros(1, dtype = torch.float32);
        L2_Loss_Value       = torch.zeros(1, dtype = torch.float32);
        Total_Loss_Value    = torch.zeros(1, dtype = torch.float32);

        # First, calculate the Lp loss, since it is not specific to each data set.
        Lp_Loss_Value = Lp_Loss(    Xi      = Xi,
                                    Mask    = Mask,
                                    p       = p);
        Lp_Loss_Buffer = Lp_Loss_Value.detach().item();

        # Now calculate the losses for each data set.
        for i in range(Num_DataSets):
            # Get the collocation, data, and L2 loss for the ith data set.
            ith_Weak_Form_Loss_Value, ith_Residual = Weak_Form_Loss(
                                            U                   = U_List[i],
                                            Xi                  = Xi,
                                            Mask                = Mask,
                                            LHS_Term            = LHS_Term,
                                            RHS_Terms           = RHS_Terms,
                                            Weight_Functions    = Weight_Functions_List[i]);

            ith_Data_Loss_Value = Data_Loss(U                   = U_List[i],
                                            Inputs              = Inputs_List[i],
                                            Targets             = Targets_List[i]);

            ith_L2_Loss_Value = L2_Squared_Loss(U = U_List[i]);

            ith_Total_Loss_Value = (Weights["Data"]*ith_Data_Loss_Value + 
                                    Weights["Weak"]*ith_Weak_Form_Loss_Value + 
                                    Weights["Lp"]*Lp_Loss_Value + 
                                    Weights["L2"]*ith_L2_Loss_Value);

            # Store those losses in the buffers (for the returned dict)
            Residual_List[i][:] = ith_Residual.detach();
            Weak_Loss_List[i]   = ith_Weak_Form_Loss_Value.detach().item();
            Data_Loss_List[i]   = ith_Data_Loss_Value.detach().item();
            L2_Loss_List[i]     = ith_L2_Loss_Value.detach().item();
            Total_Loss_List[i]  = ith_Total_Loss_Value.detach().item();

            # Finally, accumulate the losses.
            Weak_Loss_Value     += ith_Weak_Form_Loss_Value;
            Data_Loss_Value     += ith_Data_Loss_Value;
            L2_Loss_Value       += ith_L2_Loss_Value;
            Total_Loss_Value    += ith_Total_Loss_Value;

        # Back-propagate to compute gradients of Total_Loss with respect to
        # network parameters (only do if this if the loss requires grad)
        if (Total_Loss_Value.requires_grad == True):
            Total_Loss_Value.backward();

        return Total_Loss_Value;

    # update network parameters.
    Optimizer.step(Closure);

    # Return the residual tensor.
    return {"Residuals"     : Residual_List,
            "Weak Losses"   : Weak_Loss_List,
            "Data Losses"   : Data_Loss_List,
            "Lp Loss"       : Lp_Loss_Buffer,
            "L2 Losses"     : L2_Loss_List,
            "Total Losses"  : Total_Loss_List};



def Testing(
        U_List                              : List[Network],
        Xi                                  : torch.Tensor,
        Mask                                : torch.Tensor,
        Inputs_List                         : List[torch.Tensor],
        Targets_List                        : List[torch.Tensor],
        LHS_Term                            : Library_Term,
        RHS_Terms                           : List[Library_Term],
        Weight_Functions_List               : List[List[Weight_Function]],
        p                                   : float,
        Weights                             : Dict[str, float],
        Device                              : torch.device = torch.device('cpu')) -> Tuple[float, float]:
    """ 
    This function evaluates the losses.

    ----------------------------------------------------------------------------
    Arguments:

    U_List: A list of Networks whose ith element holds the network that 
    approximates the ith PDE solution.
    
    Xi: The vector that stores the coefficients of the library terms.

    Mask: A boolean tensor whose shape matches that of Xi. We use this to 
    compute the Weak Form loss (see that doc string).

    Inputs_List: A list of tensors whose ith element holds the coordinates of 
    the points at which we compare U_List[i] to the ith true solution. If each 
    U_List[i] accepts d spatial coordinates, then this should be a d+1 column 
    tensor whose kth row holds the t, x_1,... x_d coordinates of the kth 
    Data-point for U_List[i].

    Targets_List: A list of Tensors whose ith entry holds the value of the ith 
    true solution at Inputs_List[i]. If Inputs_List[i] has N rows, then this 
    should be an N element tensor of floats whose kth element holds the value 
    of the ith true solution at the kth row of Inputs_List[k].

    Time_Derivative_Order: We try to solve a PDE of the form (d^n U/dt^n) =
    N(U, D_{x}U, ...). This is the 'n' on the left-hand side of that PDE.

    LHS_Term: The Library Term (trial function + derivative) that appears on
    the left hand side of the PDE. This is generally a time derivative of U.

    RHS_Terms: A list of the Library terms (trial function + derivative)
    that appear in the right hand side of the PDE.

    Weight_Functions: A list of the weight functions in the weak form loss.

    p, Lambda: the settings value for p and Lambda (in the loss function).
    
    Weights: A dictionary of floats. It should have keys for "Lp", "Coll", and 
    "Data".

    Device: The device for the U_i's and Xi.

    ----------------------------------------------------------------------------
    Returns:

    A dictionary with the following keys:
        "Coll Loss", "Data Loss", "L2 Loss": lists of floats whose ith entry
        holds the corresponding loss for the ith data set. 

        "Total Loss": a list of floats whose ith entry houses the total loss for
        the ith data set.

        "Lp Loss": A float housing the value of the Lp loss.
    """

    assert(len(U_List) == len(Weight_Functions_List));
    assert(len(U_List) == len(Inputs_List));
    assert(len(U_List) == len(Targets_List));
    
    Num_DataSets : int = len(U_List);

    # Put each U in evaluation mode
    for i in range(Num_DataSets):
        U_List[i].eval();
    
    # First, evaluate the Lp loss, since this does not depend on the data set.
    Lp_Loss_Value : float = Lp_Loss(    Xi    = Xi,
                                        Mask  = Mask,
                                        p     = p).item();

    # Get the losses for each data set.
    Data_Loss_List  : List[float] = [0]*Num_DataSets;
    Weak_Loss_List  : List[float] = [0]*Num_DataSets;
    L2_Loss_List    : List[float] = [0]*Num_DataSets;
    Total_Loss_List : List[float] = [0]*Num_DataSets;

    for i in range(Num_DataSets):
        Data_Loss_List[i] = Data_Loss(  U           = U_List[i],
                                        Inputs      = Inputs_List[i],
                                        Targets     = Targets_List[i]).item();

        Weak_Loss_List = Weak_Form_Loss(    U                   = U_List[i],
                                            Xi                  = Xi,
                                            Mask        = Mask,
                                            LHS_Term            = LHS_Term,
                                            RHS_Terms           = RHS_Terms,
                                            Mask                = Mask,
                                            Weight_Functions    = Weight_Functions_List[i])[0].item();

        L2_Loss_List[i] = L2_Squared_Loss(U = U_List[i]).item();

        Total_Loss_List[i] =          ( Weights["Data"]*Data_Loss_List[i] + 
                                        Weights["Weak"]*Weak_Loss_List[i] + 
                                        Weights["Lp"]*Lp_Loss_Value + 
                                        Weights["L2"]*L2_Loss_List[i]);

    # Return the losses.
    return {"Data Losses"   : Data_Loss_List,
            "Weak Losses"   : Weak_Loss_List,
            "Lp Loss"       : Lp_Loss_Value,
            "L2 Losses"     : L2_Loss_List,
            "Total Losses"  : Total_Loss_List};
