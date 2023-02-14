import  torch;
from    typing          import Dict, List;

from    File_Reader     import Read_Error, Read_Line_After, Read_Bool_Setting, Read_Setting, Read_List_Setting, Read_Dict_Setting;
from    Library_Reader  import Read_Library;


def Settings_Reader() -> Dict:
    """ 
    This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Settings_Container object that contains all the settings we read from
    Settings.txt. The main function uses these to set up the program. 
    """

    File        = open("../Settings.txt", "r");
    Settings    = {};


    ############################################################################
    # Save, Load Settings

    # Load Sol, Xi, or Optimizer from File?
    Settings["Load U"]              = Read_Bool_Setting(File, "Load U's From Save [bool]:");
    Settings["Load Xi, Library"]    = Read_Bool_Setting(File, "Load Xi, Library from Save [bool]:");
    Settings["Load Optimizer"]      = Read_Bool_Setting(File, "Load Optimizer from Save [bool]:");

    # If so, get the load file name.
    if( Settings["Load U"]              == True or
        Settings["Load Xi, Library"]    == True or
        Settings["Load Optimizer"]      == True):

        Settings["Load File Name"] = Read_Line_After(File, "Load File Name [str]:").strip();



    ############################################################################
    # Library Settings.

    # If we are not loading the library from save, then where is the file that 
    # lists the library functions / derivatives?
    if(Settings["Load Xi, Library"] == False):
        Library_File_Name : str     = Read_Setting(File, "Library File [str]:");
        Settings["Library Path"]    = "../" + Library_File_Name + ".txt";



    ############################################################################
    # Network settings.

    # Read in network architecture if we are not loading U from save.
    if(Settings["Load U"] == False):
        # Width of the hidden layers
        Buffer : List[str] = Read_List_Setting(File, "Hidden Layer Widths [List of int]:");
        for i in range(len(Buffer)):
            Buffer[i] = int(Buffer[i]);
        
        Settings["Hidden Layer Widths"] = Buffer;

        # Which activation function should we use?
        Buffer = Read_Setting(File, "Activation Function [Tanh, Rational, Sin]:");
        if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
            Settings["Hidden Activation Function"] = "Rational";
        elif(Buffer[0] == 'T' or Buffer[0] == 't'):
            Settings["Hidden Activation Function"] = "Tanh";
        elif(Buffer[0] == 'S' or Buffer[0] == 's'):
            Settings["Hidden Activation Function"] = "Sin";
        else:
            raise Read_Error("\"Activation Function [Tanh, Rational, Sin]:\" should be" + \
                            "\"Tanh\", \"Rational\", or \"Sin\" Got " + Buffer);

    # Read the device.
    Buffer = Read_Setting(File, "Train on CPU or GPU [GPU, CPU]:");
    if(Buffer[0] == 'G' or Buffer[0] == 'g'):
        if(torch.cuda.is_available() == True):
            Settings["Device"] = torch.device('cuda');
        else:
            Settings["Device"] = torch.device('cpu');
            print("You requested a GPU, but cuda is not available on this machine. Switching to CPU");
    elif(Buffer[0] == 'C' or Buffer[0] == 'c'):
        Settings["Device"] = torch.device('cpu');
    else:
        raise Read_Error("\"Train on CPU or GPU\" should be \"CPU\" or \"GPU\". Got " + Buffer);



    ############################################################################
    # Loss settings.

    # Read p values (used in the Lp loss function).
    Settings["p"]      = float(Read_Setting(File, "p [float]:"));

    # Read weights (used to scale the components of the loss function)
    Buffer : Dict[str, str] = Read_Dict_Setting(File, "Weights [Dict of float]:");

    for (key, value) in Buffer.items():
        Buffer[key] = float(value);
    Settings["Weights"] = Buffer;

    # Read the number of weight functions
    Settings["Num Weight Functions"] = int(Read_Setting(File, "Number of Weight Functions [int]:"));

    # Read the Axis Partition Size. Suppose the problem domain is [a_1, b_1] x
    # ... x [a_n, b_n]. If the Axis Partition size is N, then we partition each
    # [a_i, b_i] using a uniform partition with N points. We then use these axis
    # partitions to define a partition of the entire. This is the set of
    # points of the form (p_1, ... , p_n) where each p_i is in an element of
    # the partition for the ith axis.
    Settings["Axis Partition Size"] = int(Read_Setting(File, "Axis Partition Size [int]:"));

    # Read in if we should mask small components of Xi.
    Settings["Mask Small Xi Components"] = Read_Bool_Setting(File, "Mask Small Xi Components [bool]:");


    ############################################################################
    # Optimizer settings.

    # Read the optimizer type.
    Buffer = Read_Setting(File, "Optimizer [Adam, LBFGS]:");
    if  (Buffer[0] == 'A' or Buffer[0] == 'a'):
        Settings["Optimizer"] = "Adam";
    elif(Buffer[0] == 'L' or Buffer[0] == 'l'):
        Settings["Optimizer"] = "LBFGS";
    else:
        raise Read_Error("\"Optimizer [Adam, LBFGS]:\" should be \"Adam\" or \"LBFGS\". Got " + Buffer);

    # Read the learning rate, number of epochs.
    Settings["Learning Rate"] = float(Read_Setting(File, "Learning Rate [float]:"));
    Settings["Num Epochs"]    = int(  Read_Setting(File, "Number of Epochs [int]:"));
    


    ############################################################################
    # Data settings.

    # Read in the list of datasets, assuming we are not loading U from file. 
    # Otherwise, the data set names should have been saved in the save sate.
    if(Settings["Load U"] == False):
        Settings["DataSet Names"] = Read_List_Setting(File, "DataSet Names [List of str]:");

    # All done! Return the settings!
    File.close();
    return Settings;
