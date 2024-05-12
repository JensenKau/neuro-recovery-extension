from __future__ import annotations

from rest_framework.generics import ListAPIView, CreateAPIView
from rest_framework.request import Request
from rest_framework.response import Response 

from ..serializers import PredictionSerializer
from ..models import Prediction

class GetPrediction(ListAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def get(self, request: Request) -> Response:
        data = request.query_params
        
        eeg_id = data["eeg_id"]
        
        query = Prediction.objects.get(patient_eeg_id=eeg_id)
        serializer = PredictionSerializer(query)
        
        return Response(serializer.data)
    
    
class UpdatePredictionComment(CreateAPIView):
    def get_queryset(self):
        return super().get_queryset()
    
    def post(self, request: Request) -> Response:
        user = request.user
        data = request.data
                
        if user.role != "doctor":
            return Response({"message": "Unauthorized Access"}, status=500)
                
        eeg_id = data["eeg_id"]
        comment = data["comment"]
        
        query = Prediction.objects.get(id=eeg_id)
        query.comments = comment
        query.save()
        
        return Response({"message": "Change Successful"})


if __name__ == "__main__":
    pass