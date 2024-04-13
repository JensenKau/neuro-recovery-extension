from __future__ import annotations

from django.db import models
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager

class UserManager(BaseUserManager):
    def create_user(self, email: str, firstname: str, lastname: str, fullname: str, password: str = None) -> User:
        if not email:
            raise ValueError('Users must have an email address')
        
        email = email.strip().lower()
        firstname = firstname.strip()
        lastname = lastname.strip()
        fullname = fullname.strip()
        
        user = self.model(
            email=self.normalize_email(email),
            firstname=firstname,
            lastname=lastname,
            fullname=fullname
        )
        
        user.set_password(password)
        user.save()
        
        return user
        
    
    def create_superuser(self, email: str, firstname: str, lastname: str, fullname: str, password: str = None) -> User:
        user = self.create_user(
            email,
            firstname,
            lastname,
            fullname,
            password=password,
        )
        
        user.is_superuser = True
        user.save()
        
        return user


class User(AbstractBaseUser):
    email = models.CharField(primary_key=True, max_length=155)
    
    firstname = models.CharField(max_length=100)
    lastname = models.CharField(max_length=100)
    fullname = models.CharField(max_length=255)
    
    is_active = models.BooleanField(default=True)
    is_superuser = models.BooleanField(default=False)
    
    role = models.CharField(
        max_length=20, 
        choices={
            "doctor": "doctor",
            "normal": "normal"
        },
        default="normal"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    USERNAME_FIELD = "email"
    REQUIRED_FIELDS = [
        "firstname",
        "lastname",
        "fullname"
    ]
    
    object = UserManager()
    
    
    def __str__(self) -> str:
        return self.email
    



if __name__ == "__main__":
    pass