from __future__ import annotations

from django.db import models

class AiModel(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()
    filename = models.CharField(max_length=255)
    
    
if __name__ == "__main__":
    pass