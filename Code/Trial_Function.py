


class Trial_Function():
    """ Objects of this class hold an abstract representation of a trial.

    ----------------------------------------------------------------------------
    Members:

    Power: Currently, trial functions are restricted to the form U^p, for some
    p > 0. The Power member specifies the value p. Thus, if T is a
    Trial_Function object, and T.Power = 5, then T represents the function U^5. """

    def __init__(   self,
                    Power : int) -> None:
        """ Initializer:

        ------------------------------------------------------------------------
        Arguments:

        Power: See class docstring. """

        assert(Power > 0);
        self.Power = Power;

    def Print(self) -> None:
        """ This function prints a human-readable form of the object's trial
        function """

        if(self.Power == 1):
            print("U", end = '');
        else:
            print("U^%u" % self.Power, end = '');
