# Nonsense to add Readers directory to the Python search path.
import os
import sys

# Get path to Reader directory.
Main_Path       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));
Code_Path       = os.path.join(Main_Path, "Code");
Readers_Path    = os.path.join(Code_Path, "Readers");

# Add the Readers, Classes directories to the python path.
sys.path.append(Readers_Path);

import  torch;
from    typing      import List, Dict;

from    File_Reader import Read_Error, Read_List_Setting, Read_Bool_Setting, Read_Setting;



def Settings_Reader() -> Dict:
    """
    This function reads the settings in Settings.txt.

    ---------------------------------------------------------------------------
    Arguments:

    None!

    ---------------------------------------------------------------------------
    Returns:

    A Settings_Container object that contains all the settings we read from
    Settings.txt. The main function uses these to set up the program. 
    """

    # Open file, initialize a Settings object.
    File        = open("Settings.txt", "r");
    Settings    = {};

    # Where is the saved state?
    Settings["Load File Name"]      = Read_Setting(File, "Load File Name [str]:");

    # Read the Data file names. Note that the data files should NOT contain 
    # noise.
    Settings["Mat File Names"]      = Read_List_Setting(File, "Mat File Names [List of str]:");

    # Read if we should make transparent plots.
    Settings["Transparent Plots"]    = Read_Bool_Setting(File, "Transparent Plots [bool]:")

    # All done! Return the settings!
    File.close();
    return Settings;
