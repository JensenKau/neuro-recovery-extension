from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 
from django.db.models import Q

from ..models import Patient, User
from ..serializers import ShortPatientSerializer, PatientSerializer


class GetPatientView(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request):
        data = request.query_params
                
        patient = Patient.objects.get(id=data["id"])
        patient_serializer = PatientSerializer(patient)
        
        return Response(patient_serializer.data)




class GetPatientsView(ListAPIView):    
    def get_queryset(self):
        return Patient.objects.none()
    
    def get(self, request: Request):
        user = request.user
        user_email = user.email
        
        owned = Patient.objects.filter(owner__email=user_email)
        
        access = None
        if user.role == "doctor":
            access = Patient.objects.filter(~Q(owner_id=user_email))
        else:
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
        
        if data["firstname"] == "" or data["lastname"] == "":
            return Response({"message: blank firstname or lastname"}, status=500)
        
        new_patient = Patient.objects.create(
            owner_id = user_email,
            name = f"{data['firstname']} {data['lastname']}",
            first_name = data["firstname"],
            last_name = data["lastname"],
            age = data["age"],
            sex = data["gender"],
            rosc = data["rosc"],
            ohca = data["ohca"],
            shockable_rhythm = data["sr"],
            ttm = data["ttm"]
        )
        
        return Response(ShortPatientSerializer(new_patient).data) 
    



class AddUserAccess(CreateAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        data = request.data
                
        patient_id = data["patient_id"]
        email = data["email"]
        
        Patient.access.through.objects.create(
            patient_id=patient_id,
            user_id=email
        )
        
        return Response({})
    
    
    

class DeleteUserAccess(CreateAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request):
        data = request.data
        
        patient_id = data["patient_id"]
        email = data["email"]
        
        Patient.access.through.objects.filter(
            patient_id=patient_id,
            user_id=email
        ).delete()
        
        return Response({})


if __name__ == "__main__":
    pass