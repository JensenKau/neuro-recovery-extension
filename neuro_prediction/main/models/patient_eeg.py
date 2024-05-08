from __future__ import annotations

from django.db import models
from django.contrib.postgres.fields import ArrayField

from .patient import Patient

class PatientEEG(models.Model):
    name = models.CharField(max_length=255)
    patient = models.ForeignKey(Patient, on_delete=models.CASCADE)
    
    start_time = models.PositiveBigIntegerField()
    end_time = models.PositiveBigIntegerField()
    utility_freq = models.PositiveSmallIntegerField()
    sampling_freq = models.PositiveSmallIntegerField()
    
    raw_file = models.FileField(upload_to="raw_eeg")
    proc_file = models.FileField(upload_to="proc_eeg")
    
    static_fc = ArrayField(ArrayField(models.FloatField(), 22), 22)
    avg_fc = ArrayField(ArrayField(models.FloatField(), 22), 22)
    std_fc = ArrayField(ArrayField(models.FloatField(), 22), 22)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ("name", "patient")
    


if __name__ == "__main__":
    pass