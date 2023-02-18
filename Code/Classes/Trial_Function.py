from    typing  import Dict;



class Trial_Function():
    """ 
    Objects of this class hold an abstract representation of a trial.

    ---------------------------------------------------------------------------
    Members:

    Power: Currently, trial functions are restricted to the form U^p, for some
    p >= 0. The Power member specifies the value p. Thus, if T is a
    Trial_Function object, and T.Power = 5, then T represents the function U^5. 
    """

    def __init__(   self,
                    Power : int) -> None:
        """ 
        Initializer:

        -----------------------------------------------------------------------
        Arguments:

        Power: See class docstring. 
        """

        assert(Power >= 0);
        self.Power = Power;



    def __str__(self) -> str:
        """ 
        This function returns a string that contains a human-readable form of 
        the object's trial function.
        """

        Buffer : str = "";

        # Either list 1 (if Power = 0), U (if Power = 1), or U^Powers
        # (if Power > 1).
        if  (self.Power == 0):
            Buffer += "1"
        elif(self.Power == 1):
            Buffer += "U";
        else:
            Buffer += ("U^%u" % self.Power);

        # All done. Return!
        return Buffer;
    

    def Get_State(self) -> Dict: 
        """
        This function helps serialize self. It returns a dictionary that can be
        used to create self from scratch. You can recover a copy of self by 
        passing this dictionary to the Build_Trial_Function_From_State function.
        
        -----------------------------------------------------------------------
        Returns:

        A dictionary with one key: "Power". 
        """   

        return {"Power" : self.Power}; 



def Build_Trial_Function_From_State(State : Dict) -> Trial_Function:
    """
    This function builds a new Trial Function object from a State dictionary. 
    It then returns that object. 

    ---------------------------------------------------------------------------
    Arguments:

    State: A dictionary. This should either be the dictionary returned by the 
    Trial_Function class' Get_State method, or an unpickled copy of one. 

    ---------------------------------------------------------------------------
    Returns:

    A new Trial Function object.
    """

    return Trial_Function(State["Power"]);