import mlmodels
import patientdata

from mlmodels.svm_model import SVMModel
import random

if __name__ == "__main__":
    x_list = list(x for x in range(20))
    y_list = list(random.randint(0, 1) for _ in range(20))
    
    model = SVMModel()
    
    print(model.get_data_split(x_list, y_list))