from __future__ import annotations

from django.urls import path, include

from .views import *


user_urls = [
    path("create_user/", CreateUserView.as_view()),
]

patient_urls = [
    path("get_patient/", GetPatientView.as_view()),
    path("get_patients/", GetPatientsView.as_view()),
    path("create_patient/", CreatePatientView.as_view()),
    path("add_user/", AddUserAccess.as_view()),
    path("delete_user/", DeleteUserAccess.as_view()),
]

main_urls = [
    path("user/", include(user_urls)),
    path("patient/", include(patient_urls)),
]
 
if __name__ == "__main__":
    pass