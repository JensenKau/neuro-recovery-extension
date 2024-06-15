from __future__ import annotations
from enum import Enum

class PatientSex(Enum):
    MALE = 1
    FEMALE = 2
    NONE = 0
        
class PatientOutcome(Enum):
    GOOD = 1
    POOR = 2
    NONE = 0