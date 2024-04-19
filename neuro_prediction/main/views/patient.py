from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 

from ..models import Patient, User
from ..serializers import ShortPatientSerializer, PatientSerializer

class GetPatientsView(ListAPIView):    
    def get_queryset(self):
        return Patient.objects.none()
    
    def get(self, request: Request):
        user_email = request.user.email
        
        owned = Patient.objects.filter(owner__email=user_email)
        access = Patient.objects.filter(access__email=user_email)
                
        owned_serializer = ShortPatientSerializer(owned, many=True)
        access_serializer = ShortPatientSerializer(access, many=True) 
        
        return Response({
            "owned": owned_serializer.data,
            "access": access_serializer.data
        })

if __name__ == "__main__":
    pass