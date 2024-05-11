from __future__ import annotations

from django.urls import path, include

from .views import *


user_urls = [
    path("create_user/", CreateUserView.as_view()),
    path("patient_access/", PatientAccess.as_view()),
]

patient_urls = [
    path("get_patient/", GetPatientView.as_view()),
    path("get_patients/", GetPatientsView.as_view()),
    path("create_patient/", CreatePatientView.as_view()),
    path("add_user/", AddUserAccess.as_view()),
    path("delete_user/", DeleteUserAccess.as_view()),
]

patient_eeg_urls = [
    path("generate_eeg/", GenerateEEGData.as_view()),
    path("get_eegs/", GetEEGs.as_view()),
    path("get_eeg/", GetEEG.as_view()),
    path("get_fcs/", GetFCs.as_view()),
    path("get_eeg_points/", GetEEGPoints.as_view()),
    path("get_brain_plot/", GetBrainPlot.as_view()),
]

prediction_urls = [
    path("get_prediction/", GetPrediction.as_view()),
    path("update_comment/", UpdatePredictionComment.as_view()),
]

main_urls = [
    path("user/", include(user_urls)),
    path("patient/", include(patient_urls)),
    path("patient_eeg/", include(patient_eeg_urls)),
    path("prediction/", include(prediction_urls)),
]
 
if __name__ == "__main__":
    pass