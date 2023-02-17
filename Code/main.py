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

from    Settings_Reader     import Settings_Reader;
from    Library_Reader      import Read_Library;

from    Network             import Rational, Network;
from    Weight_Function     import Weight_Function;
from    Library_Term        import Library_Term, Build_Library_Term_From_State;
from    Derivative          import Derivative;

from    Data                import Data_Loader;
from    Points              import Generate_Points, Setup_Partition;
from    Test_Train          import Testing, Training;
from    Plot                import Plot_Losses;



Threshold : float = 0.0005;



def main():
    # Load the settings, print them.
    Settings : Dict = Settings_Reader();
    for (Setting, Value) in Settings.items():
        print("%-25s = %s" % (Setting, str(Value)));

    # Start a setup timer.
    Setup_Timer : float = time.perf_counter();
    print("\nSetting up...\n");



    ############################################################################
    # Load the saved state, if we need it.

    # First, if we are loading anything, load in the save.
    if( Settings["Load U's"]            == True or 
        Settings["Load Xi, Library"]    == True or
        Settings["Load Optimizer"]      == True):

        # Load the saved checkpoint. Make sure to map it to the correct device.
        Load_File_Path : str = "../Saves/" + Settings["Load File Name"];
        Saved_State = torch.load(Load_File_Path, map_location = Settings["Device"]);



    ############################################################################
    # Set up Data
    # This sets up the testing/training inputs/targets, and the input bounds.
    # This will also tell us the number of spatial dimensions (there's a row of
    # Input_Bounds for each coordinate component. Since one coordinate is for
    # time, one minus the number of rows gives the number of spatial dimensions).

    # Load the data set names if they were saved. 
    if(Settings["Load U's"] == True):
        Settings["DataSet Names"] = Saved_State["DataSet Names"];

    Num_DataSets    : int                       = len(Settings["DataSet Names"]);
    Data_Dict       : Dict[str, numpy.ndarray]  = { "Train Inputs"          : [],
                                                    "Train Targets"         : [],
                                                    "Test Inputs"           : [],
                                                    "Test Targets"          : [],
                                                    "Input Bounds"          : [],
                                                    "Number of Dimensions"  : []};
    for i in range(Num_DataSets):
        ith_Data_Dict : Dict = Data_Loader( DataSet_Name    = Settings["DataSet Names"][i],
                                            Device          = Settings["Device"]);
        
        Data_Dict["Train Inputs"            ].append(ith_Data_Dict["Train Inputs"]);
        Data_Dict["Train Targets"           ].append(ith_Data_Dict["Train Targets"]);
        Data_Dict["Test Inputs"             ].append(ith_Data_Dict["Test Inputs"]);
        Data_Dict["Test Targets"            ].append(ith_Data_Dict["Test Targets"]);
        Data_Dict["Input Bounds"            ].append(ith_Data_Dict["Input Bounds"]);
        Data_Dict["Number of Dimensions"    ].append(ith_Data_Dict["Input Bounds"].shape[0]);

    # Now determine the number of dimensions of the data. This should be the 
    # same for each data set.
    Num_Dimensions : int   = Data_Dict["Number of Dimensions"][0];
    for i in range(1, Num_DataSets):
        assert(Data_Dict["Number of Dimensions"][i] == Num_Dimensions);



    ############################################################################
    # Set up U for each data set, as well as the common Xi, Library, and mask.

    # First, either build U or load its state from save. 
    if(Settings["Load U's"] == True):
        # Fetch each solution's state. There should be one state for each data set.
        U_States : List[Dict] = Saved_State["U States"];
        assert(len(U_States) == Num_DataSets);

        U_List : List[Network] = [];
        for i in range(Num_DataSets):
            # First, fetch Widths and Activation functions from U[i]'s State.
            Ui_State            : Dict      = U_States[i];
            Widths              : List[int] = Ui_State["Widths"];
            Hidden_Activation   : str       = Ui_State["Activation Types"][0];
            Output_Activation   : str       = Ui_State["Activation Types"][-1];

            Settings["Hidden Activation Function"]  = Hidden_Activation;

            # Set up U[i].
            U_List.append(Network(  Widths              = Widths, 
                                    Hidden_Activation   = Hidden_Activation, 
                                    Output_Activation   = Output_Activation,
                                    Device              = Settings["Device"]));
            U_List[i].Set_State(Ui_State);

            # Report!
            print("Loaded U[%u] from state." % i);
            print("    Hidden Activation:     %s" % Hidden_Activation);
            print("    Widths:                %s\n" % str(Widths));

    else:
        # First, set up Widths. This is an array whose ith entry specifies the width of 
        # the ith layer of the network (including the input and output layers).
        Widths = [Num_Dimensions] + Settings["Hidden Layer Widths"] + [1];

        # Now initialize each U[i].
        U_List : List[Network] = [];
        for i in range(Num_DataSets):
            U_List.append(Network(  Widths              = Widths,
                                    Hidden_Activation   = Settings["Hidden Activation Function"],
                                    Output_Activation   = "None",
                                    Device              = Settings["Device"]));
        
        print("Set up the solution networks using settings in Settings.txt.")


    # Second, either build Xi + library or load it from save. Also build the mask.
    if(Settings["Load Xi, Library"] == True):
        # First, load Xi.
        Xi : torch.Tensor = Saved_State["Xi"];

        # Next, load the library (LHS term and RHS Terms)
        Settings["LHS Term"] = Build_Library_Term_From_State(Saved_State["LHS Term State"]);

        RHS_Terms       : List[Library_Term]    = [];
        Num_RHS_Terms   : int                   = len(Saved_State["RHS Term States"]);
        for i in range(Num_RHS_Terms):
            RHS_Terms.append(Build_Library_Term_From_State(Saved_State["RHS Term States"][i]));
        Settings["RHS Terms"] = RHS_Terms;     

        print("Loaded Xi, Library from file.");

    else:
        # First, read the library.
        LHS_Term, RHS_Terms     = Read_Library(Settings["Library Path"]);
        Settings["LHS Term"]    = LHS_Term;
        Settings["RHS Terms"]   = RHS_Terms;

        # Next, determine how many library terms we have. This determines Xi's 
        # size.
        Num_RHS_Terms : int = len(Settings["RHS Terms"]);

        # Since we want to learn Xi, we set its requires_Grad to true.
        Xi = torch.zeros(   Num_RHS_Terms,
                            dtype           = torch.float32,
                            device          = Settings["Device"],
                            requires_grad   = True);
                    
        print("Build Xi, Library using settings in Settings.txt");
    
    # Make a copy of Xi. We will use this after training to counter momentum 
    # (see below)
    Initial_Xi = torch.clone(Xi);

    # Report!
    print("    Xi:                    [", end = '');
    for k in range(Num_RHS_Terms):
        print("%.5f" % Xi[k].item(), end = '');
        if(k == Num_RHS_Terms - 1):
            print(" ]");
        else:
            print(", ", end = '');

    print("    LHS Term:              %s" % str(Settings["LHS Term"]))
    print("    RHS Terms (%3u total): " % Num_RHS_Terms, end = '');
    for i in range(Num_RHS_Terms):
        print(str(Settings["RHS Terms"][i]), end = '');
        
        if(i != Num_RHS_Terms - 1):
            print(", ", end = '');
        else:
            print("\n");

    # Build the mask.
    Mask : torch.Tensor = torch.zeros(Num_RHS_Terms, dtype = torch.bool);
    if(Settings["Load Xi, Library"] == True and Settings["Mask Small Xi Components"] == True):
        for i in range(Num_RHS_Terms):
            if(abs(Xi[i].item()) < Threshold):
                Mask[i] = True;   
    print("Masking %u RHS terms\n" % torch.sum(Mask));



    ############################################################################
    # Set up the optimizer.
    # Note: we need to do this after loading Xi, since loading Xi potentially
    # overwrites the original Xi (loading the optimizer later ensures the
    # optimizer optimizes the correct Xi tensor).

    Params = [];
    for i in range(Num_DataSets):
        Params = Params + list(U_List[i].parameters());
    Params.append(Xi);

    if(  Settings["Optimizer"] == "Adam"):
        Optimizer = torch.optim.Adam( Params,   lr = Settings["Learning Rate"]);
    elif(Settings["Optimizer"] == "LBFGS"):
        Optimizer = torch.optim.LBFGS(Params,   lr = Settings["Learning Rate"]);
    else:
        print(("Optimizer is %s when it should be \"Adam\" or \"LBFGS\"" % Settings["Optimizer"]));
        exit();

    if(Settings["Load Optimizer"]  == True ):
        # Now load the optimizer.
        Optimizer.load_state_dict(Saved_State["Optimizer"]);

        # Enforce the new learning rate (do not use the saved one).
        for param_group in Optimizer.param_groups:
            param_group['lr'] = Settings["Learning Rate"];

    # Setup is now complete. Report time.
    print("Set up complete! Took %7.2fs\n" % (time.perf_counter() - Setup_Timer));
    


    ############################################################################
    # Set up weight functions.

    print("Reminder: Replace the \"Setup Weight Functions\" code with something better...\n");

    Weight_Functions_List : List[List[Weight_Function]] = [];
    for i in range(Num_DataSets):
        # Get the bounds for the ith data set
        ith_Bounds      : numpy.ndarray = Data_Dict["Input Bounds"][i];
        ith_Partition   : torch.tensor  = Setup_Partition(  Axis_Partition_Size = Settings["Axis Partition Size"],
                                                            Bounds              = ith_Bounds);

        # Set up the volume for the ith data set (and report).
        ith_V : float = 1;
        for j in range(Num_Dimensions):
            ith_V *= (ith_Bounds[j, 1] - ith_Bounds[j, 0])/float(Settings["Axis Partition Size"]);
        
        # Report problem domain, sub-rectangle volume.
        print("Problem domain %d:" % i); 
        print("\tBounds               - ", end = '');

        for j in range(Num_Dimensions):
            print("[%g, %g]" % (Data_Dict["Input Bounds"][i][j][0], Data_Dict["Input Bounds"][i][j][1]), end = '');
            if(j != Num_Dimensions - 1):
                print(" x ", end = '');
            else:
                print();
        print("\tSub-rectangle volume - %f" % ith_V);

        # Set up weight functions for the ith data set.
        ith_Min_Side_Length : float = ith_Bounds[0, 1] - ith_Bounds[0, 0];

        for i in range(1, Num_Dimensions):
            if(ith_Bounds[i, 1] - ith_Bounds[i, 0] < ith_Min_Side_Length):
                ith_Min_Side_Length = ith_Bounds[i, 1] - ith_Bounds[i, 0];

        # Set up radius.... Maybe make this better?
        ith_Radius  : float = ith_Min_Side_Length*(.4);

        # Set up weight function centers.
        # If the problem domain is [a_1, b_1] x ... x [a_n, b_n], then we place the
        # centers in [a_1 + r + e, b_1 - r - e] x ... x [a_n - r + e, b_n - r - e],
        # where e = Epsilon is some small positive number (to ensure the weight
        # function support is in the domain).
        Epsilon             : float         = 0.0005;
        Trimmed_ith_Bounds  : numpy.ndarray = numpy.empty_like(ith_Bounds);

        for i in range(Num_Dimensions):
            Trimmed_ith_Bounds[i, 0] = ith_Bounds[i, 0] + ith_Radius + Epsilon;
            Trimmed_ith_Bounds[i, 1] = ith_Bounds[i, 1] - ith_Radius - Epsilon;

        # Generate Centers.
        ith_Centers : numpy.ndarray = Generate_Points(
                                    Bounds     = Trimmed_ith_Bounds,
                                    Num_Points = Settings["Num Weight Functions"],
                                    Device     = Settings["Device"]);

        # Set up the weight functions for the ith data set
        ith_Weight_Functions : List[Weight_Function] = [];
        for j in range(Settings["Num Weight Functions"]):
            w_j = Weight_Function(
                        X_0     = ith_Centers[j],
                        r       = ith_Radius,
                        Coords  = ith_Partition,
                        V       = ith_V);

            ith_Weight_Functions.append(w_j);

        # Compute weight function derivatives.
        for j in range(Settings["Num Weight Functions"]):
            w_j : Weight_Function = ith_Weight_Functions[j];

            # First, add the LHS Term derivative.
            w_j.Add_Derivative(Settings["LHS Term"].Derivative);

            # Now add the derivatives from the RHS Terms.
            for k in range(Num_RHS_Terms):
                w_j.Add_Derivative(Settings["RHS Terms"][k].Derivative);

        # Add the weight functions to the list.
        Weight_Functions_List.append(ith_Weight_Functions);



   ############################################################################
    # Run the Epochs!

    # Set up buffers to hold the losses.
    Train_Losses        : List[Dict[str, numpy.ndarray]]    = [];
    Test_Losses         : List[Dict[str, numpy.ndarray]]    = [];
    L2_Losses           : List[numpy.ndarray]               = [];
    Lp_Losses           : numpy.ndarray                     = numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32);

    for i in range(Num_DataSets):
        Train_Losses.append({   "Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Weak Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)});

        Test_Losses.append({    "Data Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Weak Losses"    : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32),
                                "Total Losses"   : numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32)});

        L2_Losses.append(numpy.ndarray(shape = Settings["Num Epochs"], dtype = numpy.float32));

    # Epochs!!!
    Epoch_Timer         : float                             = time.perf_counter();
    print("\nRunning %d epochs..." % Settings["Num Epochs"]);
    for t in range(0, Settings["Num Epochs"]):
        ########################################################################
        # Train

        # Run one epoch of training. 
        Train_Dict = Training(  U_List                  = U_List,
                                Xi                      = Xi,
                                Mask                    = Mask,
                                Inputs_List             = Data_Dict["Train Inputs"],
                                Targets_List            = Data_Dict["Train Targets"],
                                LHS_Term                = Settings["LHS Term"],
                                RHS_Terms               = Settings["RHS Terms"],
                                Weight_Functions_List   = Weight_Functions_List,
                                p                       = Settings["p"],
                                Weights                 = Settings["Weights"],
                                Optimizer               = Optimizer,
                                Device                  = Settings["Device"]);

        # Append the train loss history.
        for i in range(Num_DataSets):
            Train_Losses[i]["Data Losses"][t]  = Train_Dict["Data Losses"][i];
            Train_Losses[i]["Weak Losses"][t]  = Train_Dict["Weak Losses"][i];
            Train_Losses[i]["Total Losses"][t] = Train_Dict["Total Losses"][i];



        ########################################################################
        # Test

        # Evaluate losses on the testing points.
        Test_Dict = Testing(    U_List                  = U_List,
                                Xi                      = Xi,
                                Mask                    = Mask,
                                Inputs_List             = Data_Dict["Test Inputs"],
                                Targets_List            = Data_Dict["Test Targets"],
                                LHS_Term                = Settings["LHS Term"],
                                RHS_Terms               = Settings["RHS Terms"],
                                Weight_Functions_List   = Weight_Functions_List,
                                p                       = Settings["p"],
                                Weights                 = Settings["Weights"],
                                Device                  = Settings["Device"]);

        # Append the test loss history.
        for i in range(Num_DataSets):
            Test_Losses[i]["Data Losses"][t]   = Test_Dict["Data Losses"][i];
            Test_Losses[i]["Weak Losses"][t]   = Test_Dict["Weak Losses"][i];
            Test_Losses[i]["Total Losses"][t]  = Test_Dict["Total Losses"][i];

        # Now record the Lp, L2 losses.
        Lp_Losses[t] = Test_Dict["Lp Loss"];

        for i in range(Num_DataSets):
            L2_Losses[i][t] = Test_Dict["L2 Losses"][i];



        ########################################################################
        # Report!

        if(t % 10 == 0 or t == Settings["Num Epochs"] - 1):
            for i in range(Num_DataSets):
                print("            |");
                print("            | Train:\t Data[%u] = %.7f\t Weak[%u] = %.7f\t Total[%u] = %.7f" % (i, Train_Dict["Data Losses"][i], i, Train_Dict["Weak Losses"][i], i, Train_Dict["Total Losses"][i]));
                print("            | Test: \t Data[%u] = %.7f\t Weak[%u] = %.7f\t Total[%u] = %.7f" % (i, Test_Dict["Data Losses"][i], i, Test_Dict["Weak Losses"][i], i, Test_Dict["Total Losses"][i]));
                if(i == 0):
                    print("Epoch #%-4d |       \t Lp      = %.7f\t L2[%u]   = %.7f" % (t + 1, Test_Dict["Lp Loss"], i, Test_Dict["L2 Losses"][i]));
                else: 
                    print("            |       \t Lp      = %.7f\t L2[%u]   = %.7f" % (Test_Dict["Lp Loss"], i, Test_Dict["L2 Losses"][i]));
                print("            |");                
        else:
            print("Epoch #%-4d |" % (t + 1));

    # Finally, replaced the final masked components of Xi with their 
    # pre-training values.  Why do we do this? It's complicated.... Some
    # optimizers have momentum. This means that even if the derivative of the
    # loss function with respect to a parameter is zero, the optimizer may 
    # still update the value of that parameter. This creates a big problem for
    # masked RHS terms. If a RHS term is masked, we do not want its component
    # of Xi to change. The collocation and Lp loss functions are built such
    # that if a term is masked, the corresponding component of Xi will have NO
    # impact on the loss (so the derivative of the loss function with respect
    # to that component of Xi is zero).... but momentum still allows the 
    # parameter to change. To fix this, replace the final masked components of 
    # Xi with their original ones. 
    Xi.requires_grad_(False);
    for k in range(Num_RHS_Terms):
        if(Mask[k] == True):
            Xi[k] = Initial_Xi[k];
    Xi.requires_grad_(True);

    # Report runtime!
    Epoch_Runtime : float = time.perf_counter() - Epoch_Timer;
    print("Done! It took %7.2fs, an average of %7.2fs per epoch)" % (Epoch_Runtime,  (Epoch_Runtime / Settings["Num Epochs"])));



    ############################################################################
    # Report final PDE.

    # Print the LHS Term.
    print();
    print(Settings["LHS Term"], end = '');
    print(" = ", end = '');

    # Print the RHS terms, with thresholding. 
    # Recall how we enforce Xi: We trick torch into minimizing
    #       [Xi]_1^p + ... + [Xi]_n^p,
    # which is highly concave (for p < 1), by instead minimizing
    #       w_1[Xi]_1^2 + ... + w_n[Xi]_n^2,
    # where each w_i is updated each step to be w_i = [Xi]_i^{2 - p}. The above
    # is convex (if we treat w_1, ... , w_n as constants). There is, however, a
    # problem. If [Xi]_i is smaller than about 3e-4, then [Xi]_i^2 is roughly
    # machine Epsilon, meaning we run into problems. To avoid this, we instead
    # define
    #       w_i = max{1e-7, [Xi]_i^{p - 2}}.
    # The issue with this approach is that the Lp loss can't really resolve
    # components of Xi which are smaller than about 3e-4. To deal with this, we
    # ignore all components smaller than the threshold (which should be set to 
    # be slightly larger than 3e-4)
    Num_Terms_Printed : int = 0;
    for k in range(len(Settings["RHS Terms"])):
        # Skip if the term if masked, or thresholded.
        if(Mask[k] == True):
            continue;
        elif(abs(Xi[k].item()) < Threshold):
            continue;
        
        if(  Num_Terms_Printed != 0 and Xi[k] > 0):
            print(" + ", end = '');
        elif(Xi[k] < 0):
            print(" - ", end = '');
        print("%7.4f" % torch.abs(Xi[k]), end = '');

        print(Settings["RHS Terms"][k], end = '');
        Num_Terms_Printed += 1;
    # End the line.
    print();



    ############################################################################
    # Save.

    print("\nSaving...", end = '');

    # First, come up with a save name that does not conflict with an existing
    # save name. To do this, we first attempt to make a save file name that
    # consists of the data set name plus the activation function and optimizer
    # we used. If a save with that name already exists, we append a "1" to the
    # end of the file name. If that also corresponds to an existing save, then
    # we replace the "1" with a "2" and so on until we get save name that does
    # not already exist.
    Base_File_Name  : str = "";
    for i in range(Num_DataSets):
        Base_File_Name += Settings["DataSet Names"][i] + "_";
    Base_File_Name  += Settings["Hidden Activation Function"] + "_" + Settings["Optimizer"];

    Counter         : int = 0;
    Save_File_Name  : str = Base_File_Name;
    while(os.path.isfile("../Saves/" + Save_File_Name)):
        # Increment the counter, try appending that onto Base_File_Name.
        Counter         += 1;
        Save_File_Name   = Base_File_Name + ("_%u" % Counter);

    # Next, get each U's state
    U_States : List[Dict] = [];
    for i in range(Num_DataSets):
        U_States.append(U_List[i].Get_State());

    # Next, get the state of each library term.
    RHS_Term_States : List[Dict] = [];
    for i in range(len(Settings["RHS Terms"])):
        RHS_Term_States.append(Settings["RHS Terms"][i].Get_State());
    
    LHS_Term_State : Dict = Settings["LHS Term"].Get_State();    

    # We can now save!
    torch.save({"U States"              : U_States,
                "Xi"                    : Xi,
                "Optimizer"             : Optimizer.state_dict(),
                "LHS Term State"        : LHS_Term_State,
                "RHS Term States"       : RHS_Term_States, 
                "DataSet Names"         : Settings["DataSet Names"]},
                "../Saves/" + Save_File_Name);

    print("Done! Saved as \"%s\"" % Save_File_Name);



    ############################################################################
    # Plot. 

    Plot_Losses(Save_File_Name      = Save_File_Name,
                Train_Losses        = Train_Losses,
                Test_Losses         = Test_Losses,
                L2_Losses           = L2_Losses,
                Lp_Losses           = Lp_Losses,
                Labels              = Settings["DataSet Names"]);
                


if(__name__ == "__main__"):
    main();
