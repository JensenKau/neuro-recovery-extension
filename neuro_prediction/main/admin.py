from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(AiModel)
admin.site.register(PatientEEG)
admin.site.register(Patient)
admin.site.register(Prediction)
admin.site.register(User)