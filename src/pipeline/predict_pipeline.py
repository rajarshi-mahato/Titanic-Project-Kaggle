import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(
        self,

        # Numerical Features
        Age: float,
        RoomService: float,
        FoodCourt: float,
        ShoppingMall: float,
        Spa: float,
        VRDeck: float,

        # Categorical Features
        HomePlanet: str,
        CryoSleep: str,
        Destination: str,
        VIP: str
    ):

        # Numerical
        self.Age = Age
        self.RoomService = RoomService
        self.FoodCourt = FoodCourt
        self.ShoppingMall = ShoppingMall
        self.Spa = Spa
        self.VRDeck = VRDeck

        # Categorical
        self.HomePlanet = HomePlanet
        self.CryoSleep = CryoSleep
        self.Destination = Destination
        self.VIP = VIP

    def get_data_as_data_frame(self):
        try:

            custom_data_input_dict = {

                # Numerical Features
                "Age": [self.Age],
                "RoomService": [self.RoomService],
                "FoodCourt": [self.FoodCourt],
                "ShoppingMall": [self.ShoppingMall],
                "Spa": [self.Spa],
                "VRDeck": [self.VRDeck],

                # Categorical Features
                "HomePlanet": [self.HomePlanet],
                "CryoSleep": [self.CryoSleep],
                "Destination": [self.Destination],
                "VIP": [self.VIP]

            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)