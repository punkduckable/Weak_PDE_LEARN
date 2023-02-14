from    typing          import Dict, List;

import  Derivative;
import  Trial_Function;




class Library_Term():
    """ 
    This class represents a Library term (derivative + trial function).

    ----------------------------------------------------------------------------
    Members:

    Derivative: A Derivative object that houses the derivative portion of this
    library term (this may be the identity).

    Trial_Function: A Trial_Function object that houses the trial function
    portion of this library term. 
    """

    def __init__(   self,
                    Derivative      : Derivative.Derivative,
                    Trial_Function  : Trial_Function.Trial_Function) -> None:
        """ 
        Initializer.

        ------------------------------------------------------------------------
        Arguments:

        Derivative, Trial_Function: See class docstring. 
        """

        self.Derivative     = Derivative;
        self.Trial_Function = Trial_Function;



    def __str__(self) -> str:
        """ 
        This function returns a human-readable form of the Library term that 
        this object represents.
        """

        Buffer : str = "(";

        # Get the Derivative.
        Buffer += self.Derivative.__str__();

        # Get the Trial Function.
        Buffer += self.Trial_Function.__str__() + ")";

        # All done! Return!
        return Buffer;



    def Get_State(self) -> Dict:
        """
        This function helps serialize self. It returns a dictionary that can be
        used to create self from scratch. You can recover a copy of self by 
        passing this dictionary to the Build_Library_Term_From_State function.
        
        -----------------------------------------------------------------------
        Returns:

        A dictionary with two keys, "Derivative Encodings" and "Powers". The 
        former is a list whose ith entry holds the Encoding array for the ith 
        derivative of the library term. The latter is simply self's power 
        attribute.
        """

        # We can start things off with the Powers attribute.
        State : Dict = {"Trial Function"    : self.Trial_Function.Get_State(),
                        "Derivative"        : self.Derivative.Get_State()};

        return State;




def Build_Library_Term_From_State(State : Dict) -> Library_Term:
    """
    This function builds a new Library Term object from a State dictionary. It then
    returns that object. 

    ---------------------------------------------------------------------------
    Arguments:

    State: A dictionary. This should either be the dictionary returned by the 
    Library Term class' Get_State method, or an unpickled copy of one. 

    ---------------------------------------------------------------------------
    Returns:

    A new Library Term object. 
    """

    # Extract the derivative, trial function.
    D : Derivative.Derivative           = Derivative.Build_Derivative_From_State(State["Derivative"]);
    F : Trial_Function.Trial_Function   = Trial_Function.Build_Trial_Function_From_State(State["Trial Function"]);

    # Now... build the library term
    return Library_Term(Derivative = D, Trial_Function = F);