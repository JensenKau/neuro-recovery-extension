from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 
from rest_framework.permissions import AllowAny

from ..models import User


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
        
        User.objects.create_user(email, firstname, lastname, fullname, password)
        
        return Response({"email": email})




if __name__ == "__main__":
    pass