import torch;
import numpy;
import math;

from Evaluate_Derivatives import Evaluate_Derivatives;
from Network import Neural_Network;


print("If you're reading this, then Robert forgot to write the printing...");


"""
def Print_PDE(  Xi                     : torch.Tensor,
                Time_Derivative_Order  : int,
                Num_Spatial_Dimensions : int,
                Index_to_Derivatives,
                Col_Number_to_Multi_Index):
    This function prints out the PDE encoded in Xi. Suppose that Xi has
    N + 1 components. Then Xi[0] - Xi[N - 1] correspond to PDE library terms,
    while Xi[N] correponds to a constant. Given some k in {0,1,... ,N-1} we
    first map k to a multi-index (using Col_Number_to_Multi_Index). We then map
    each sub-index to a spatial partial derivative of x. We then print out this
    spatial derivative.

    ----------------------------------------------------------------------------
    Arguments:

    Xi: The Xi tensor. If there are N library terms (Col_Number_to_Multi_Index.
    Total_Indices = N), then this should be an N+1 component tensor.

    Time_Derivative_Order: We try to solve a PDE of the form (d^n U/dt^n) =
    N(U, D_{x}U, ...). This is the 'n' on the left-hand side of that PDE.

    Num_Spatial_Dimensions: The number of spatial dimensions in the underlying
    data set. We need this to construct the library terms.

    Index_to_Derivatives: If Num_Spatial_Dimensions = 1, then this maps
    sub-index value to a number of x derivatives. If Num_Spatial_Dimensions = 2,
    then this maps a sub-index value to a number of x and y derivatives.

    Col_Number_to_Multi_Index: This maps column numbers (library term numbers)
    to Multi-Indices.

    ----------------------------------------------------------------------------
    Returns:

    Nothing :D


    if(Time_Derivative_Order == 1):
        print("D_t U = ");
    else:
        print("D_t^%u U = " % Time_Derivative_Order);


    N : int = Xi.numel();
    for k in range(0, N - 1):
        # Fetch the kth component of Xi.
        Xi_k = Xi[k].item();

        # If it's non-zero, fetch the associated multi-Inde
        if(Xi_k == 0):
            continue;
        Multi_Index = Col_Number_to_Multi_Index(k);

        # Cycle through the sub-indices, printing out the associated derivatives
        print("+ %7.4f" % Xi_k, end = '');
        Num_Indices = Multi_Index.size;

        for j in range(0, Num_Indices):
            if  (Num_Spatial_Dimensions == 1):
                Num_x_Deriv : int = Index_to_Derivatives(Multi_Index[j].item());
                if(Num_x_Deriv == 0):
                    print("(U)", end = '');
                else:
                    print("(D_x^%d U)" % Num_x_Deriv, end = '');

            elif(Num_Spatial_Dimensions == 2):
                Num_x_Deriv, Num_y_Deriv = Index_to_Derivatives(Multi_Index[j].item());
                if(Num_x_Deriv == 0):
                    if(Num_y_Deriv == 0):
                        print("(U)", end = '');
                    else:
                        print("(D_y^%d U)" % Num_y_Deriv, end = '');
                elif(Num_y_Deriv == 0):
                    print("(D_x^%d U)" % Num_x_Deriv, end = '');
                else:
                    print("(D_x^%d D_y^%d U)" % (Num_x_Deriv, Num_y_Deriv), end = '');
        print("");

    # Now print out the constant term.
    if(Xi[N - 1] != 0):
        print("+ %7.4f" % Xi[N - 1].item());
"""
