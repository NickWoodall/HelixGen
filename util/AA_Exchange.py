from weakref import WeakKeyDictionary

class AAProperty(object):
    ## dictionary mapping one letter code to amino acid
    AA = {"K": "Lys", "R": "Arg", "D": "Asp", "E": "Glu", "Q": "Gln", "N": "Asn", \
          "T": "Thr", "S": "Ser", "H": "His", "Y": "Tyr", "W": "Trp", "F": "Phe", \
          "M": "Met", "C": "Cys", "L": "Leu", "I": "Ile", "V": "Val", "A": "Ala", "G": "Gly",
          "P": "Pro", " ":" "}  ##dictonary for aminoAcids

    ## create reverse dictionary by iterating through each key an value
    revAA = dict()
    for key, val in AA.items():
        revAA[val] = key

    fullAA = {"Lysine": "K", "Arginine": "R", "Aspartate": "D", "Glutamate": "E", "Glutamine": "Q", "Asparagine": "N", \
              "Threonine": "T", "Serine": "S", "Histidine": "H", "Tyrosine": "Y", "Tryptophan": "W",
              "Phenylalanine": "F", \
              "Methionine": "M", "Cysteine": "C", "Leucine": "L", "Isoleucine": "I", "Valine": "V", "Alanine": "G",
              "Proline": "P", " ": " "}

    revFullAA = dict()
    for key, val in fullAA.items():
        revFullAA[val] = key

    ## uses weaky key dictionary to allow python to garbage collect ref from
    ## each descriptor references from the values dictionary

    def __init__(self):
        self.values = WeakKeyDictionary()

    ## set to the one letter code if the it matches the amino acid case insenstive or not
    def __set__(self, obj, val):

        if len(val) == 1:  # one letter code check
            val = val.upper()
            if val in self.AA:  ## is the value in the dictionary
                self.values[obj] = val.upper()
            else:
                raise AttributeError("Not an Amino Acid")
        elif len(val) == 3:  ## three letter code check
            theKey = val[0:1].upper() + val[1:3].lower()  ## convert to case in dictionary
            if theKey in self.revAA:
                self.values[obj] = self.revAA[theKey]  ## sets to values dictionary for static AAproperty
            else:
                raise AttributeError("Not an Amino Acid")
        else:
            # check against full name, store as single letter code
            theKey = val[0:1].upper() + val[1:].lower()
            if theKey in self.fullAA:
                self.values[obj] = self.fullAA[theKey]

    def __get__(self, obj, objtype):

        return self.values.get(obj)  ## return value for this instance

    def __delete__(self, obj):
        del self.values[obj]  ## delete value for this instance



class AminoAcid(object):
    x = AAProperty();  ## static declaration need dicionary (values to keep track of instances )

    def __init__(self, name1="R"):
        self.x = name1


def aaCodeExchange(aaIn, typeOut="one"):

    aa = AminoAcid(aaIn)
    ## dictionary mapping one letter code to amino acid
    AA = {"K": "Lys", "R": "Arg", "D": "Asp", "E": "Glu", "Q": "Gln", "N": "Asn", \
          "T": "Thr", "S": "Ser", "H": "His", "Y": "Tyr", "W": "Trp", "F": "Phe", \
          "M": "Met", "C": "Cys", "L": "Leu", "I": "Ile", "V": "Val", "A": "Ala", "G": "Gly",
          "P": "Pro", " " : " "}  ##dictonary for aminoAcids

    ## create reverse dictionary by iterating through each key an value
    revAA = dict()
    for key, val in AA.items():
        revAA[val] = key

    fullAA = {"Lysine": "K", "Arginine": "R", "Aspartate": "D", "Glutamate": "E", "Glutamine": "Q", "Asparagine": "N", \
          "Threonine": "T", "Serine": "S", "Histidine": "H", "Tyrosine": "Y", "Tryptophan": "W",
          "Phenylalanine": "F", \
          "Methionine": "M", "Cysteine": "C", "Leucine": "L", "Isoleucine": "I", "Valine": "V", "Alanine": "G",
          "Proline": "P", " " : " "}

    revFullAA = dict()
    for key, val in fullAA.items():
        revFullAA[val] = key

    if typeOut == "one" :
        return aa.x
    elif typeOut =="three":
        return AA[aa.x]
    elif typeOut =="threeCAP":
        return AA[aa.x].upper()
    elif typeOut =="full":
        return revFullAA[aa.x]

    return -1