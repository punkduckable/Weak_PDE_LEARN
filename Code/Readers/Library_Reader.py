# Nonsense to add Code dirctory to the Python search path.
import os
import sys

# Get path to parent directory (the Code directory, in this case)
Code_path  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)));

# Add the Code directory to the python path.
sys.path.append(Code_path);

import  numpy;
from    typing          import List, Tuple;

from    File_Reader     import End_Of_File_Error, Read_Line_After;
from    Derivative      import Derivative;
from    Trial_Function  import Trial_Function;
from    Library_Term    import Library_Term;






def Parse_Trial_Function(Buffer : str):
    """ 
    This function reads a Trial Function from the string "Buffer".

    ----------------------------------------------------------------------------
    Arguments:

    Buffer: This should be a string that contains a representation of the
    Library term we want to read. In general, this will be of the form
        S_1(U)*S_2(U)*...*S_K(U)
    Currently, we require each S_i(U) to be either 1, U, or U^p, for some p > 1.
    We may weaken this constraint in the future. There should be no whitespace
    in this expression. 
    """

    Power : int = 0;

    # First, split the Buffer along the '*' character
    Components      : List[str] = Buffer.split('*');
    Num_Components  : int       = len(Components);

    # We now read in each S_i(U)
    for i in range(Num_Components):
        # Recall that S_i(U) is either U or U^p, for some p > 0. We handle these
        # cases separately.
        if  ('^' in Components[i]):
            Power += int(Components[i][-1]);
        elif(Components[i] == "1"):
            Power += 0;
        elif(Components[i] == "U"):
            Power += 1;

    # Now that we know the power, form the Trial Function and return.
    F : Trial_Function = Trial_Function(Power = Power);
    return F;



def Parse_Library_Term(Buffer : str) -> Library_Term:
    """ 
    This function parses a library term from a line of the Library file.
    The "Buffer" argument should be a stripped line of Library.txt that contains
    a library term. In general, Read_Library_Term is the only function that
    should call this one. We read the library term within, and then process it.

    ----------------------------------------------------------------------------
    Arguments:

    Buffer: A stripped line containing a library term (derivative + trial
    function). 
    """

    # First, split the line by spaces.
    Components : List[str] = Buffer.split(" ");

    # Second, parse the trial function (which should be the last component).
    F : Trial_Function = Parse_Trial_Function(Components[-1]);

    # Next, parse the derivatives. If there is no derivative, then we can build
    # the Library Term.
    Num_Derivative_Terms : int = len(Components) - 1;

    if(Num_Derivative_Terms == 0):
        Encoding    : numpy.ndarray = numpy.array([0, 0], dtype = numpy.int32);
        D           : Derivative    = Derivative(Encoding = Encoding);

        Term : Library_Term = Library_Term( Derivative      = D,
                                            Trial_Function  = F);

        return Term;

    # Otherwise, parse the derivatives. We'll start with an encoding for a
    # derivative with three spatial variables, then trim the unused dimensions
    # once we're done parsing.
    Encoding : numpy.ndarray = numpy.zeros(shape = (4), dtype = numpy.int32);
    for i in range(Num_Derivative_Terms):
        # First, send everything to lower case (this makes processing easier).
        Component : str = Components[i].lower();

        # First, split at the '_'. The first character after this contains
        # the variable.
        Component : str = Component.split('_')[1];

        j : int = 0;
        if  (Component[0] == 't'):
            j = 0;
        elif(Component[0] == 'x'):
            j = 1;
        elif(Component[0] == 'y'):
            j = 2;
        elif(Component[0] == 'z'):
            j = 3;
        else:
            raise Read_Error("Derivative term has the wrong format. Buffer = "  + Buffer);

        # Each derivative term should either be of the form D_s or D_s^p. We
        # handle these cases separately.
        p : int = 1;
        if  ('^' in Component):
            p : int = int(Component[-1]);
        Encoding[j] += p;

    # Now, trim down the encoding (if possible).
    if(Encoding[3] == 0):
        Encoding = Encoding[0:3];

        if(Encoding[2] == 0):
            Encoding = Encoding[0:2];

    D    : Derivative   = Derivative(Encoding = Encoding);
    Term : Library_Term = Library_Term( Derivative      = D,
                                        Trial_Function  = F);

    return Term;



def Read_Library_Term(File):
    """ 
    This function reads a library term (derivative + trial function) from the 
    Library file. To do that, we search through the file for the first line 
    that is neither blank nor entirely a comment. We then parse the term 
    within.

    ---------------------------------------------------------------------------
    Arguments:

    File: The file we want to read a library term from. This file should 
    contain library terms (as strings) using the format specified in 
    Library.txt. 
    """

    # Look for the next line that contains a library function.
    Line : str = "";
    while(True):
        # Get a candidate line. This eliminates all lines that start with a
        # comment or are blank. It will not, however, eliminate lines filled
        # with whitespace.
        Line = Read_Line_After( File    = File,
                                Phrase  = "");


        # Strip. If the line contains only whitespace, this will reduce it to
        # an empty string. If this is the case, move onto the next line.
        # Otherwise, the line should contain a library term.
        Line = Line.strip();
        if(len(Line) == 0):
            continue;
        else:
            break;

    # Now turn it into a Library Term object.
    Term : Library_Term = Parse_Library_Term(Line);
    return Term;



def Read_Library(File_Path : str) -> Tuple[Library_Term, List[Library_Term]]:
    """ 
    This function reads the Library terms in Library.txt.

    ----------------------------------------------------------------------------
    Arguments:

    File_Path: This is the name (relative to the working director of the file
    that houses the main function that called this one, and with a .txt
    extension of the library file). Thus, if we run this from Code/main.txt, and
    the Library function is Library.txt, then File_Path should be ../Library.txt
    """

    # First, open the file.
    File = open(File_Path, 'r');

    # Next, read the LHS Term. This is the first Library term in the file.
    LHS_Term : Library_Term = Read_Library_Term(File);

    # Finally, read the RHS Terms.
    RHS_Terms : List[Library_Term] = [];
    while(True):
        try:
            Term : Library_Term = Read_Library_Term(File);
        except End_Of_File_Error:
            # If we raise this exception, then we're done.
            break;
        else:
            # Otherwise, add the new term to the LHS_Terms list.
            RHS_Terms.append(Term);

    # All done!
    File.close();
    return LHS_Term, RHS_Terms;



def main():
    File_Path : str = "../../Library.txt";
    RHS_Term, LHS_Terms = Read_Library(File_Path = File_Path);

    print(RHS_Term);
    for i in range(len(LHS_Terms)):
        print(LHS_Terms[i]);


if __name__ == "__main__":
    main();
