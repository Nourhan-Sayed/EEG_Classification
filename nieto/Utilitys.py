# -*- coding: utf-8 -*-

"""
@author: Nicolás Nieto - nnieto@sinc.unl.edu.ar

Utilitys for Inner speech dataset prossesing
"""

def Ensure_dir(dir_name):
    import os
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        

def picks_from_channels(channels):
    """
    Parameters
    ----------
    channels : str
        Name of the channel or regions.

    Returns
    -------
    picks : list
        List of picks that corresponds with the channels.

    """
    if channels == "A":
        picks = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16","A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32"]
    elif channels == "B":
        picks = ["B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","B13","B14","B15","B16","B17","B18","B19","B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30","B31","B32"]
    elif channels == "C":
        picks = ["C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16","C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C27","C28","C29","C30","C31","C32"]
    elif channels == "D":
        picks = ["D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16","D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"]
    
    elif channels == "OCC_L" or channels =="OL":
        picks = ["A10","A8","D30","A9"]
    elif channels == "OCC_Z" or channels == "OZ":
        picks = ["A22","A23","A24","A15","A28"]
    elif channels == "OCC_R" or channels =="OR":
        picks = ["B12","B5","B6","B7"]

    elif channels == "FRONT_L" or channels =="FL":
        picks = ["D6","D5","C32","C31"]
    elif channels == "FRONT_Z" or channels =="FZ":
        picks = ["C18","C20","C27","C14"]
    elif channels == "FRONT_R" or channels =="FR":
        picks = ["C9","C6","C10","C5"]

    elif channels == "C_L" or channels =="CL":
        picks = ["D26","D21","D10","D19"]
    elif channels == "C_Z" or channels =="CZ":
        picks = ["D15","A1","B1","A2"]
    elif channels == "C_R" or channels =="CR":
        picks = ["B16","B24","B29","B22"]
        
    elif channels == "P_Z" or channels =="PZ":
        picks = ["A4","A19","A20","A32","A5"]    
        
    elif channels == "OP_Z" or channels =="OPZ":
        picks = ["A17","A30","A20","A21","A22"]    
         
    elif channels =="all" or channels=="All":
          picks = "all"  
    else:
        picks = []
        print("Invalid channels name")
        
    return picks


def unify_names(Cond, Class):
    
    if Cond == "inner" or Cond == "In":
        Cond = "Inner"
        
    if Cond == "vis" or Cond == "Visualized":
        Cond = "Vis"
        
    if Cond == "pron" or Cond == "Pronounced":
        Cond = "Pron"
        
    if Class == "all" or Class == "Todo" or Class == "todo":
        Class = "All"
        
    if Class == "up" or Class == "Arriba" or Class == "arriba":
        Class = "Up"
        
    if Class == "down" or Class == "Abajo" or Class == "abajo":
        Class = "Down"
        
    if Class == "right" or Class == "Derecha" or Class == "derecha":
        Class = "Right"

    if Class == "left" or Class == "Izquierda" or Class == "izquierda":
        Class = "Left"
        
    return Cond, Class

def map_condition(cnd):
    if not cnd:
        raise Exception("Condition is empty!")

    if cnd.upper() == "A" or cnd.upper() == "ALL":
        return "ALL"
    if cnd.upper() == "P" or cnd.upper() == "PRON" or cnd.upper() == "PRONOUNCED":
        return "PRONOUNCED"
    if cnd.upper() == "I" or cnd.upper() == "IN" or cnd.upper() == "INNER":
        return "INNER"
    if cnd.upper() == "V" or cnd.upper() == "VIS" or cnd.upper() == "VISUALIZED":
        return "VISUALIZED"

    raise Exception("Wrong name of condition!")

def map_class(cls):
    if not cls:
        raise Exception("Class is empty!")

    if cls.upper() == "ALL" or cls.upper() == "TODOS":
        return "ALL"
    if cls.upper() == "U" or cls.upper() == "UP" or cls.upper() == "AR" or cls.upper() == "ARRIBA":
        return "UP"
    if cls.upper() == "D" or cls.upper() == "DOWN" or cls.upper() == "AB" or cls.upper() == "ABAJO":
        return "DOWN"
    if cls.upper() == "L" or cls.upper() == "LEFT" or cls.upper() == "I" or cls.upper() == "IZQ" or cls.upper() == "IZQUIERDA":
        return "LEFT"
    if cls.upper() == "R" or cls.upper() == "RIGHT" or cls.upper() == "D" or cls.upper() == "DER" or cls.upper() == "DERECHA":
        return "RIGHT"

    raise Exception("Wrong name of class!")    
