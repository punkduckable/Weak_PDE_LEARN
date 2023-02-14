# Nonsense to add Readers, Classes directories to the Python search path.
import os
import sys

# Get path to Code, Readers, Classes directories.
Code_Path       = os.path.dirname(os.path.abspath(__file__));
Readers_Path    = os.path.join(Code_Path, "Readers");
Classes_Path    = os.path.join(Code_Path, "Classes");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);
sys.path.append(Classes_Path);

import  numpy;
import  torch;
import  time;
from    typing              import Dict, List, Tuple;

from    Settings_Reader    import Settings_Reader;
from    Loss               import Data_Loss, Lp_Loss, Weak_Form_Loss;
from    Network            import Rational, Network;
from    Data_Loader        import Data_Loader;
from    Test_Train         import Testing, Training;
from    Points             import Generate_Points, Setup_Partition;
from    Weight_Function    import Weight_Function;
from    Derivative         import Derivative;



Threshold : float = 0.0005;



def main():
    # Load the settings, print them.
    Settings : Dict = Settings_Reader();
    for (Setting, Value) in Settings.items():
        print("%-25s = %s" % (Setting, str(Value)));

    # Start a setup timer.
    Setup_Timer : float = time.perf_counter();
    print("\nSetting up...\n");

    return;
    


    ############################################################################
    # Set up Data
    # This sets up the testing/training inputs/targets, and the input bounds.
    # This will also tell us the number of dimensions (there's a row of
    # Input_Bounds for each coordinate component. We can also determine the
    # highest order derivatives with respect to each variable. We will need this
    # information to set up the weight functions.

    Data_Container = Data_Loader(   DataSet_Name = Settings.DataSet_Name,
                                    Device       = Settings.Device);

    Num_RHS_Terms : int = len(Settings.RHS_Terms);

    # Get the number of input dimensions.
    Settings.Num_Dimensions : int = Data_Container.Input_Bounds.shape[0];

    # Get the highest order derivative with respect to each variable.
    Max_Derivatives : numpy.ndarray = Settings.LHS_Term.Derivative.Encoding.copy();

    for i in range(Num_RHS_Terms):
        D_i : Derivative = Settings.RHS_Terms[i].Derivative;

        for j in range(Settings.Num_Dimensions):
            if(D_i.Encoding[j] > Max_Derivatives[j]):
                Max_Derivatives[j] = D_i.Encoding[j];


    ############################################################################
    # Set up Quadrature points, Volume.
    # To do this, we place a uniform partition on the problem domain.

    Bounds      : numpy.ndarray = Data_Container.Input_Bounds;

    Partition   : torch.tensor  = Setup_Partition(  Axis_Partition_Size = Settings.Axis_Partition_Size,
                                                    Bounds              = Bounds);

    # Find volume.
    V : float = 1;
    for i in range(Settings.Num_Dimensions):
        V *= (Bounds[i, 1] - Bounds[i, 0])/float(Settings.Axis_Partition_Size);


    ############################################################################
    # Set up weight functions.
    # This is demo code. We randomly spawn weight functions in the problem
    # domain. We place the centers of each weight function such that its
    # support lies in the interior of the problem domain.

    print("Reminder: Replace the \"Setup Weight Functions\" code with something better...", end = '');

    Min_Side_Length : float = Bounds[0, 1] - Bounds[0, 0];
    for i in range(1, Settings.Num_Dimensions):
        if(Bounds[i, 1] - Bounds[i, 0] < Min_Side_Length):
            Min_Side_Length = Bounds[i, 1] - Bounds[i, 0];

    # Set up radius.
    Radius  : float = Min_Side_Length*(.4);

    # Set up weight function centers.
    # If the problem domain is [a_1, b_1] x ... x [a_n, b_n], then we place the
    # centers in [a_1 + r + e, b_1 - r - e] x ... x [a_n - r + e, b_n - r - e],
    # where e = Epsilon is some small positive number (to ensure the weight
    # function support is in the domain).
    Epsilon         : float         = 0.00001;
    Trimmed_Bounds  : numpy.ndarray  = numpy.empty_like(Bounds);

    for i in range(Settings.Num_Dimensions):
        Trimmed_Bounds[i, 0] = Bounds[i, 0] + Radius + Epsilon;
        Trimmed_Bounds[i, 1] = Bounds[i, 1] - Radius - Epsilon;

    # Generate Centers.
    Centers : numpy.ndarray = Generate_Points(
                                Bounds     = Trimmed_Bounds,
                                Num_Points = Settings.Num_Weight_Functions,
                                Device     = Settings.Device);

    # Set Powers (one more than the highest order derivative for each variable)
    Powers_np   : numpy.ndarray = numpy.add(Max_Derivatives, 1);
    Powers      : torch.Tensor  = torch.from_numpy(Powers_np);

    # Set up the weight functions.
    Weight_Functions = [];
    for i in range(Settings.Num_Weight_Functions):
        w_i = Weight_Function(
                    X_0     = Centers[i],
                    r       = Radius,
                    Coords  = Partition,
                    V       = V);

        Weight_Functions.append(w_i);


    ############################################################################
    # Compute weight function derivatives.

    for i in range(Settings.Num_Weight_Functions):
        w_i : Weight_Function = Weight_Functions[i];

        # First, add the LHS Term derivative.
        w_i.Add_Derivative(Settings.LHS_Term.Derivative);

        # Now add the derivatives from the RHS Terms.
        for j in range(Num_RHS_Terms):
            w_i.Add_Derivative(Settings.RHS_Terms[j].Derivative);


    ############################################################################
    # Set up U and Xi.

    U = Network(
            Num_Hidden_Layers   = Settings.Num_Hidden_Layers,
            Neurons_Per_Layer   = Settings.Units_Per_Layer,
            Input_Dim           = Settings.Num_Dimensions,
            Output_Dim          = 1,
            Activation_Function = Settings.Activation_Function,
            Device              = Settings.Device);

    # We set up Xi as a Parameter for.... complicated reasons. In pytorch, a
    # parameter is basically a special tensor that is supposed to be a trainable
    # part of a module. It acts just like a regular tensor, but almost always
    # has requires_grad set to true. Further, since it's a sub-class of Tensor,
    # we can distinguish it from regular Tensors. In particular, optimizers
    # expect a list or dictionary of Parameters... not Tensors. Since we want
    # to train Xi, we set it up as a Parameter.
    Xi = torch.zeros(   Num_RHS_Terms,
                        dtype           = torch.float32,
                        device          = Settings.Device,
                        requires_grad   = True);


    ############################################################################
    # Load U, Xi

    # First, check if we should load Xi, U from file. If so, load them!
    if( Settings.Load_U         == True or
        Settings.Load_Xi        == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        if(Settings.Load_U == True):
            U.load_state_dict(Saved_State["U"]);

        if(Settings.Load_Xi == True):
            Xi = Saved_State["Xi"];


    ############################################################################
    # Set up the optimizer.
    # Note: we need to do this after loading Xi, since loading Xi potentially
    # overwrites the original Xi (loading the optimizer later ensures the
    # optimizer optimizes the correct Xi tensor).

    Params = list(U.parameters());
    Params.append(Xi);

    if  (Settings.Optimizer == "Adam"):
        Optimizer = torch.optim.Adam(Params, lr = Settings.Learning_Rate);
    elif(Settings.Optimizer == "LBFGS"):
        Optimizer = torch.optim.LBFGS(Params, lr = Settings.Learning_Rate);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Settings.Optimizer));
        exit();


    if(Settings.Load_Optimizer  == True ):
        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings.Load_File_Name;
        Saved_State = torch.load(Load_File_Path, map_location = Settings.Device);

        # Now load the optimizer.
        Optimizer.load_state_dict(Saved_State["Optimizer"]);

        # Enforce the new learning rate (do not use the saved one).
        for param_group in Optimizer.param_groups:
            param_group['lr'] = Settings.Learning_Rate;


    # Setup is now complete. Report time.
    print("Done! Took %7.2fs" % (time.perf_counter() - Setup_Timer));


    ############################################################################
    # Run the Epochs!

    Epoch_Timer : float = time.perf_counter();
    print("Running %d epochs..." % Settings.Num_Epochs);

    for t in range(1, Settings.Num_Epochs + 1):
        # Run a Training Epoch.
        Training(   U                                   = U,
                    Xi                                  = Xi,
                    Inputs                              = Data_Container.Train_Inputs,
                    Targets                             = Data_Container.Train_Targets,
                    LHS_Term                            = Settings.LHS_Term,
                    RHS_Terms                           = Settings.RHS_Terms,
                    Partition                           = Partition,
                    V                                   = V,
                    Weight_Functions                    = Weight_Functions,
                    p                                   = Settings.p,
                    Lambda                              = Settings.Lambda,
                    Optimizer                           = Optimizer,
                    Device                              = Settings.Device);

        # Test the code (and print the loss) every 10 Epochs. For all other
        # epochs, print the Epoch to indicate the program is making progress.
        if(t % 10 == 0 or t == 1):
            # Evaluate losses on training points.
            (Train_Data_Loss, Train_Coll_Loss, Train_Lp_Loss) = Testing(
                U                                   = U,
                Xi                                  = Xi,
                Inputs                              = Data_Container.Train_Inputs,
                Targets                             = Data_Container.Train_Targets,
                LHS_Term                            = Settings.LHS_Term,
                RHS_Terms                           = Settings.RHS_Terms,
                Partition                           = Partition,
                V                                   = V,
                Weight_Functions                    = Weight_Functions,
                p                                   = Settings.p,
                Lambda                              = Settings.Lambda,
                Device                              = Settings.Device);

            # Evaluate losses on the testing points.
            (Test_Data_Loss, Test_Weak_Form_Loss, Test_Lp_Loss) = Testing(
                U                                   = U,
                Xi                                  = Xi,
                Inputs                              = Data_Container.Test_Inputs,
                Targets                             = Data_Container.Test_Targets,
                LHS_Term                            = Settings.LHS_Term,
                RHS_Terms                           = Settings.RHS_Terms,
                Partition                           = Partition,
                V                                   = V,
                Weight_Functions                    = Weight_Functions,
                p                                   = Settings.p,
                Lambda                              = Settings.Lambda,
                Device                              = Settings.Device);

            # Print losses!
            print("Epoch #%-4d | Test: \t Data = %.7f\t Weak = %.7f\t Lp = %.7f \t Total = %.7f"
                % (t, Test_Data_Loss, Test_Weak_Form_Loss, Test_Lp_Loss, Test_Data_Loss + Test_Weak_Form_Loss + Test_Lp_Loss));
            print("            | Train:\t Data = %.7f\t Weak = %.7f\t Lp = %.7f \t Total = %.7f"
                % (Train_Data_Loss, Test_Weak_Form_Loss, Train_Lp_Loss, Train_Data_Loss + Test_Weak_Form_Loss + Train_Lp_Loss));
        else:
            print(("Epoch #%-4d | "   % t));

    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs," % Epoch_Runtime);
    print("an average of %7.2fs per epoch." % (Epoch_Runtime / Settings.Num_Epochs));


    ############################################################################
    # Threshold Xi.

    # Cycle through components of Xi. Remove all whose magnitude is smaller
    # than the threshold.
    Pruned_Xi = torch.empty_like(Xi);
    N   : int = Xi.numel();
    for k in range(N):
        Abs_Xi_k = abs(Xi[k].item());
        if(Abs_Xi_k < Threshold):
            Pruned_Xi[k] = 0;
        else:
            Pruned_Xi[k] = Xi[k];


    ############################################################################
    # Save.

    if(Settings.Save_State == True):
        Save_File_Path : str = "../Saves/" + Settings.Save_File_Name;
        torch.save({"U"         : U.state_dict(),
                    "Xi"        : Xi,
                    "Optimizer" : Optimizer.state_dict()},
                    Save_File_Path);


    ############################################################################
    # Report final PDE

    print(Settings.LHS_Term, end = '');
    print(" = ", end = '');

    for i in range(Num_RHS_Terms - 1):
        if(Pruned_Xi[i] != 0):
            print("+ %f*(" % Pruned_Xi[i], end = '');
            print(Settings.RHS_Terms[i], end = '');
            print(") ", end = '');

    if(Pruned_Xi[-1] != 0):
        print("+ %f*(" % Pruned_Xi[-1], end = '');
        print(Settings.RHS_Terms[-1], end = '');
        print(")", end = '');
    print();



if(__name__ == "__main__"):
    main();
