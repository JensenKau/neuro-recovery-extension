from __future__ import annotations
from enum import Enum

class PatientSex(Enum):
    MALE = 0
    FEMALE = 1
    NONE = 2
        
class PatientOutcome(Enum):
    GOOD = 0
    POOR = 1
    NONE = 2