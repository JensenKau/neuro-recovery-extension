from __future__ import annotations

from django.db import models

from .user import User

class Patient(models.Model):
    owner = models.ForeignKey(User, on_delete=models.CASCADE)
    access = models.ManyToManyField(User)
    
    name = models.CharField(max_length=255)
    age = models.PositiveIntegerField(null=True)
    sex = models.CharField(max_length=10, choices={"male": "male", "female": "female"}, null=True)
    rosc = models.FloatField(null=True)
    ohca = models.BooleanField(null=True)
    shockable_rhythm = models.BooleanField(null=True)
    ttm = models.IntegerField(null=True)
    

if __name__ == "__main__":
    pass