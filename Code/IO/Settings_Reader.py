import torch;

from File_Reader    import Read_Error, Read_Line_After;
from Library_Reader import Read_Library;



################################################################################
# Classes

class Read_Error(Exception):
    # Raised if we can't find a Phrase in a File.
    pass;

class Settings_Container:
    # A container for data read in from the settings file.
    pass;



################################################################################
# Functions



def Read_Bool_Setting(File, Setting_Name : str) -> bool:
    """ Reads a boolean setting from File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to read the setting from.

    Setting_Name: The name of the setting we're reading. We need this in case
    of an error, so that we can print the appropiate error message.

    ----------------------------------------------------------------------------
    Return:

    The value of the boolean setting. """

    # Read the setting. This will yield a string.
    Buffer = Read_Line_After(File, Setting_Name).strip();

    # Check if the setting is present. If not, the Buffer will be empty.
    if  (len(Buffer) == 0):
        raise Read_Error("Missing Setting Value: You need to specify the \"%s\" setting" % Setting_Name);

    # Attempt to parse the result. Throw an error if we can't.
    if  (Buffer[0] == 'T' or Buffer[0] == 't'):
        return True;
    elif(Buffer[0] == 'F' or Buffer[0] == 'f'):
        return False;
    else:
        raise Read_Error("Invalid Setting Value: \"%s\" should be \"True\" or \"False\". Got " % Setting_Name + Buffer);



def Read_Setting(File, Setting_Name : str) -> str:
    """ Reads a non-boolean setting from File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to read the setting from.

    Setting_Name: The name of the setting we're reading. We need this in case
    of an error, so that we can print the appropiate error message.

    ----------------------------------------------------------------------------
    Return:

    The value of the non-boolean setting as a string (you may need to type cast
    the returned value) """

    # Read the setting. This will yield a string.
    Buffer = Read_Line_After(File, Setting_Name).strip();

    # Check if the setting is present. If not, the Buffer will be empty.
    if  (len(Buffer) == 0):
        raise Read_Error("Missing Setting Value: You need to specify the \"%s\" setting" % Setting_Name);

    # Return!
    return Buffer;



def Settings_Reader() -> Settings_Container:
    """ This function reads the settings in Settings.txt.

    ----------------------------------------------------------------------------
    Arguments:

    None!

    ----------------------------------------------------------------------------
    Returns:

    A Settings_Container object that contains all the settings we read from
    Settings.txt. The main function uses these to set up the program. """

    # Open file, initialze a Settings object.
    File = open("../../Settings.txt", "r");
    Settings = Settings_Container();



    ############################################################################
    # Save, Load Settings

    # Load Sol, Xi, or Optimizer from File?
    Settings.Load_U         = Read_Bool_Setting(File, "Load U From Save [bool]:");
    Settings.Load_Xi        = Read_Bool_Setting(File, "Load Xi From Save [bool]:");
    Settings.Load_Optimizer = Read_Bool_Setting(File, "Load Optimizer From Save [bool]:");

    # If so, get the load file name.
    if( Settings.Load_U         == True or
        Settings.Load_Xi        == True or
        Settings.Load_Optimizer == True):
        Settings.Load_File_Name = Read_Line_After(File, "Load File Name [str]:").strip();

    # Check if we're saving to file.
    Settings.Save_State = Read_Bool_Setting(File, "Save State [bool]:");

    # If so, get the save file name.
    if(Settings.Save_State == True):
        Settings.Save_File_Name = Read_Setting(File, "Save File Name [str]:");



    ############################################################################
    # PDE settings.

    # The order of the time derivative on the left-hand side of the PDE.
    Settings.Time_Derivative_Order = int(Read_Setting(File, "Time Derivative Order [int]:"));

    # Where is the file that lists the library functions / derivatives?
    Library_File_Name : str =  Read_Setting(File, "Library File [str]:");
    RHS_Term, LHS_Terms     = Read_Library(Library_File_Name);

    Settings.RHS_Term       = RHS_Term;
    Settings.LHS_Terms      = LHS_Terms;



    ############################################################################
    # Network settings.

    # Number of hidden layers in U network.
    Settings.Num_Hidden_Layers = int(Read_Setting(File, "Number of Hidden Layers [int]:"));

    # Number of hidden units per hidden layer in the U network.
    Settings.Units_Per_Layer = int(Read_Setting(File, "Hidden Units per Hidden Layer [int]:"));

    # Which activation function should we use?
    Buffer = Read_Setting(File, "Activation Function [Tanh, Rational, Sin]:");
    if  (Buffer[0] == 'R' or Buffer[0] == 'r'):
        Settings.Activation_Function = "Rational";
    elif(Buffer[0] == 'T' or Buffer[0] == 't'):
        Settings.Activation_Function = "Tanh";
    elif(Buffer[0] == 'S' or Buffer[0] == 's'):
        Settings.Activation_Function = "Sin";
    else:
        raise Read_Error("\"Activation Function [Tanh, Rational, Sin]:\" should be" + \
                         "\"Tanh\", \"Rational\", or \"Sin\" Got " + Buffer);

    Buffer = Read_Setting(File, "Train on CPU or GPU [GPU, CPU]:");
    if(Buffer[0] == 'G' or Buffer[0] == 'g'):
        if(torch.cuda.is_available() == True):
            Settings.Device = torch.device('cuda');
        else:
            Settings.Device = torch.device('cpu');
            print("You requested a GPU, but cuda is not available on this machine. Switching to CPU");
    elif(Buffer[0] == 'C' or Buffer[0] == 'c'):
        Settings.Device = torch.device('cpu');
    else:
        raise Read_Error("\"Train on CPU or GPU\" should be \"CPU\" or \"GPU\". Got " + Buffer);



    ############################################################################
    # Loss settings.

    # Read p values (used in the Lp loss function).
    Settings.p      = float(Read_Setting(File, "p [float]:"));

    # Read Lambda value (used to scale the p-norm of Xi).
    Settings.Lambda = float(Read_Setting(File, "Lambda [float]:"));



    ############################################################################
    # Threshold Settings.

    Settings.Threshold = float(Read_Setting(File, "Threshold [float]:"));



    ############################################################################
    # Optimizer settings.

    # Read the optimizer type.
    Buffer = Read_Setting(File, "Optimizer [Adam, LBFGS]:");
    if  (Buffer[0] == 'A' or Buffer[0] == 'a'):
        Settings.Optimizer = "Adam";
    elif(Buffer[0] == 'L' or Buffer[0] == 'l'):
        Settings.Optimizer = "LBFGS";
    else:
        raise Read_Error("\"Optimizer [Adam, LBFGS]:\" should be \"Adam\" or \"LBFGS\". Got " + Buffer);

    # Read the learning rate, number of epochs.
    Settings.Learning_Rate = float(Read_Setting(File, "Learning Rate [float]:"));
    Settings.Num_Epochs    = int(  Read_Setting(File, "Number of Epochs [int]:"));



    ############################################################################
    # Data settings.

    # Data file name. Note that the data file should NOT contain noise.
    Settings.DataSet_Name =  Read_Setting(File, "DataSet [str]:");

    # All done! Return the settings!
    File.close();
    return Settings;
