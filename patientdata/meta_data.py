from __future__ import annotations
from enum import Enum

class PatientMetaData:
    class Sex(Enum):
        MALE = 0
        FEMALE = 1
        NONE = 2
        
    class Outcome(Enum):
        GOOD = 0
        POOR = 1
        NONE = 2
    
    def __init__(self, patient_id: int, hospital: str, age: int, sex: PatientMetaData.Sex, rosc: float, ohca: bool, shockable_rhythm: bool, ttm: int, outcome: PatientMetaData.Outcome, cpc: int) -> None:
        self.patient_id = patient_id
        self.hospital = hospital
        self.age = age
        self.sex = sex
        self.rosc = rosc
        self.ohca = ohca
        self.shockable_rhythm = shockable_rhythm
        self.ttm = ttm
        self.outcome = outcome
        self.cpc = cpc
        
    @classmethod
    def load_patient_meta_data(cls, filename: str) -> PatientMetaData:
        output = {}
        parsers = {
            "Patient": lambda x: int(x),
            "Hospital": lambda x: x,
            "Age": lambda x: int(x),
            "Sex": lambda x: PatientMetaData.Sex.MALE if x.lower() == "male" else PatientMetaData.Sex.FEMALE,
            "ROSC": lambda x: float(x),
            "OHCA": lambda x: x.lower() == "true",
            "Shockable Rhythm": lambda x: x.lower() == "true",
            "TTM": lambda x: int(x),
            "Outcome": lambda x: PatientMetaData.Outcome.GOOD if x.lower() == "good" else PatientMetaData.Outcome.POOR,
            "CPC": lambda x: int(x)
        }
        
        with open(filename, "r", encoding="utf-8") as file:            
            for line in file.read().strip().split("\n"):
                line = line.split(": ")
                output[line[0]] = parsers[line[0]](line[1])
        
        return PatientMetaData(
            patient_id=output["Patient"],
            hospital=output["Hospital"],
            age=output["Age"],
            sex=output["Sex"],
            rosc=output["ROSC"],
            ohca=output["OHCA"],
            shockable_rhythm=output["Shockable Rhythm"],
            ttm=output["TTM"],
            outcome=output["Outcome"],
            cpc=output["CPC"]
        )


if __name__ == "__main__":
    pass