class Read_Error(Exception):
    # Raised if we can't find a Phrase in a File.
    pass;



def Index_After_Phrase(
        Line_In   : str,
        Phrase_In : str,
        Case_Sensitive : bool = False) -> int:
    """ This function searches for the substring Phrase_In within Line_In.

    ----------------------------------------------------------------------------
    Arguments:

    Line_In: A string. The program searches for the substring Phrase_In within
    Line_In.

    Phrase_In: A string containing the phrase we're searching for.

    Case_Sensitive: Controls if the search is case sensitive or not. (see
    Read_Line_After's docstring for more details).

    ----------------------------------------------------------------------------
    Returns:

    If Phrase_In is a substring of Line_In, then this returns the index of the
    first character after the first instance of Phrase_In within Line_In. If
    Phrase_In is NOT a substring of Line_In, then this function returns -1. """

    # First, get the number of characters in Line/Phrase.
    Num_Chars_Line   : int = len(Line_In);
    Num_Chars_Phrase : int = len(Phrase_In);

    # If we're ignoring case, then map Phrase, Line to lower case versions of
    # themselves. Note: We don't want to modify the original variables. Since
    # Python passes by references, we store this result in a copy of Line/Phrase
    Line   = Line_In;
    Phrase = Phrase_In;
    if(Case_Sensitive == False):
        Line = Line.lower();
        Phrase = Phrase.lower();

    # If Phrase is a substring of Line, then the first character of Phrase must
    # occur in the first (Num_Chars_Line - Num_Chars_Phrase) characters of
    # Line. Thus, we only need to check the first (Num_Chars_Line -
    # Num_Chars_Phrase) characters of Line.
    for i in range(0, Num_Chars_Line - Num_Chars_Phrase - 1):
        # Check if ith character of Line matches the 0 character of Phrase. If
        # so, check if  Line[i + j] == Phrase[j] for each for each j in {0,...
        # Num_Chars_Phrase - 1}.
        if(Line[i] == Phrase[0]):
            Match : Bool = True;

            for j in range(1, Num_Chars_Phrase):
                # If Line[i + j] != Phrase[j], then we do not have a match and
                # we should move onto the next character of Line.
                if(Line[i + j] != Phrase[j]):
                    Match = False;
                    break;

            # If Match is still True, then Phrase is a substring of Line and
            # i + Num_Chars_Phrase is the index of the first character in Line
            # after Phrase.
            if(Match == True):
                return i + Num_Chars_Phrase;

    # If we're here, then Phrase is NOT a substring of Line. Return -1 to
    # indiciate that.
    return -1;



def Read_Line_After(
        File,
        Phrase         : str,
        Comment_Char   : str = '#',
        Case_Sensitive : bool = False) -> str:
    """ This function tries to find a line of File that contains Phrase as a
    substring. Note that we start searching at the current position of the file
    pointer. We do not search from the start of File.

    ----------------------------------------------------------------------------
    Arguments:

    File: The file we want to search for Phrase.

    Phrase: The Phrase we want to find. This should not include any instances
    of the comment char.

    Comment_Char: The character that denotes the start of a comment. If a line
    contains an instance of the Comment_Char, then we will remove everything in
    the line after the first instance of the Comment_Char before searching for
    a match.

    Case_Sensitive: Controls if the search is case sensitive or not. If
    True, then we search for an exact match (including case) of Phrase in one of
    File's lines. If not, then we try to find a line of File which contains the
    same letters in the same order as Phrase.

    ----------------------------------------------------------------------------
    Returns:

    If Phrase is a substring of a line of File, then this function returns
    everything in that line after the first occurrence of Phrase. If it can't
    find Phrase in one of File's lines, it raises an exception. If the Phrase is
    "cat is", and one of File's lines is "the cat is fat", then this will return
    " fat". """

    # Search the lines of File for one that contains Phrase as a substring.
    while(True):
        # Get the next line
        Line = File.readline();

        # Python doesn't use end of file characters. However, readline will
        # retun an empty string if and only if we're at the end of File. Thus,
        # we can use this as our "end of file" check
        if(Line == ""):
            raise Read_Error("Could not find \"" + Phrase + "\" in File.");

        # If the line is a comment, then ignore it.
        if (Line[0] == Comment_Char):
            continue;

        # Check for in-line comments.
        Line_Length = len(Line);
        for i in range(1, Line_Length):
            if(Line[i] == Comment_Char):
                Line = Line[:i-1];
                break;

        # Check if Phrase is a substring of Line. If so, this will return the
        # index of the first character after phrase in Line. In this case,
        # return everything in Line after that index. If Phrase is not in
        # Line. In this case, move on to the next line.
        Index : int = Index_After_Phrase(Line, Phrase, Case_Sensitive);
        if(Index == -1):
            continue;
        else:
            return Line[Index:];
