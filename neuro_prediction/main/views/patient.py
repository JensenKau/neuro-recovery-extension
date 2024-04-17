from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 

from ..models import Patient, User

class GetPatientsView(ListAPIView):    
    def get_queryset(self):
        return Patient.objects.none()
    
    def get(self, request: Request):
        user_email = request.user.email
        
        owned = Patient.objects.filter(owner__email=user_email)
        access = Patient.objects.filter(access__user_id=user_email)
        
        print(owned)
        print(access)
        
        return Response()

if __name__ == "__main__":
    pass