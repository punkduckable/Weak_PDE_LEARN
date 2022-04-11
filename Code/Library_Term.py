import Derivative;
import Trial_Function;



class Library_Term():
    """ This class represents a Library term (derivative + trial function).

    ----------------------------------------------------------------------------
    Members:

    Derivative: A Derivative object that houses the derivative portion of this
    library term (this may be the identity).

    Trial_Function: A Trial_Function object that houses the trial function
    portion of this library term. """

    def __init__(   self,
                    Derivative      : Derivative.Derivative,
                    Trial_Function  : Trial_Function.Trial_Function) -> None:
        """ Initializer.

        ------------------------------------------------------------------------
        Arguments:

        Derivative, Trial_Function: See class docstring. """

        self.Derivative     = Derivative;
        self.Trial_Function = Trial_Function;

    def Print(self):
        # Print the Derivative.
        self.Derivative.Print();

        # Print the Trial Function.
        self.Trial_Function.Print();
