from enum import Enum

class StructuralClass(str, Enum):
    PANEL = "PANEL"
    FLAP = "FLAP"
    GLUE_TAB = "GLUE_TAB"
    DUST_FLAP = "DUST_FLAP"
    BACKGROUND = "BACKGROUND"
    UNKNOWN = "UNKNOWN"
