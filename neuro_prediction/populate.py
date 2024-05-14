from __future__ import annotations

from main.models import *

try:
    User.objects.get(email="admin@fyp.com")
except User.DoesNotExist:
    User.objects.create_superuser(
        email="admin@fyp.com",
        firstname="admin",
        lastname="admin",
        fullname="admin",
        password="1234"
    )
    
try:
    AiModel.objects.get(id=1)
except AiModel.DoesNotExist:
    AiModel.objects.create(
        id=1,
        name="Model 1",
        description="some description",
        filename="some file"
    )