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

        
class CreatePatientView(CreateAPIView):
    def get_queryset(self):
        return Patient.objects.none()
    
    def post(self, request: Request):
        user_email = request.user.email
        data = request.data
        
        new_patient = Patient.objects.create(
            owner_id = user_email,
            name = f"{data['firstname']} {data['lastname']}",
            age = data["age"],
            sex = data["gender"],
            rosc = data["rosc"],
            ohca = data["ohca"],
            shockable_rhythm = data["sr"],
            ttm = data["ttm"]
        )
        
        return Response(ShortPatientSerializer(new_patient).data) 


if __name__ == "__main__":
    pass