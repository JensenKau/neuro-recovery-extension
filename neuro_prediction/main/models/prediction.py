from __future__ import annotations

from django.db import models

from .patient_eeg import PatientEEG
from .ai_model import AiModel

class Prediction(models.Model):
    patient_eeg = models.ForeignKey(PatientEEG, on_delete=models.CASCADE)
    ai_model = models.ForeignKey(AiModel, on_delete=models.CASCADE)
    
    outcome_pred = models.CharField(max_length=10, choices={"good": "good", "bad": "bad"})
    cpc_pred = models.PositiveSmallIntegerField(null=True)
    confidence = models.FloatField()

    comments = models.TextField()

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ("patient_eeg",  "ai_model")



if __name__ == "__main__":
    pass