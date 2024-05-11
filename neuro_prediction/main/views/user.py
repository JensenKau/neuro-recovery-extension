from __future__ import annotations

from django.core.validators import validate_email
from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 
from rest_framework.permissions import AllowAny

from ..models import User, Patient


class CreateUserView(CreateAPIView):
    permission_classes = [AllowAny]
    
    def get_queryset(self):
        return User.objects.none()
    
    def post(self, request: Request):
        data = request.data
        firstname = data["firstname"]
        lastname = data["lastname"]
        fullname = f"{firstname} {lastname}"
        email = data["email"]
        password = data["password"]
        
        if firstname == "" or lastname == "" or email == "" or password == "":
            return Response({}, status=500)
        
        validate_email(email)
        
        User.objects.create_user(email, firstname, lastname, fullname, password)
        
        return Response({"email": email})


class PatientAccess(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request) -> Response:
        user = request.user.email
        data = request.query_params
        
        patient_id = data["patient_id"]
        
        patient = Patient.objects.get(id=patient_id)
        access = User.objects.filter(access__id=patient_id)
                        
        for i in range(len(access)):
            if user == access[i].email:
                return Response({"access": "sucess"})
                
        if user == patient.owner_id:
            return Response({"access": "sucess"})
        
        return Response({"access": "fail"}, status=500)
    



if __name__ == "__main__":
    pass