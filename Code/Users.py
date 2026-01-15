from abc import ABC, abstractmethod
from config import Config
from DataManager import DataManager
from DataExplorer import Explorer
import numpy as np
#classe as controller
from ModelManager import ModelManager
import tkinter as tk
from tkinter import filedialog



#base class don't change
class User(ABC):
    def __init__(self, config: Config, user_id: int): #id for verification maybe???
        self.id = user_id
        self.config = config
        self.dm = DataManager(config)
        self.data = None #our image for processing

    #fragment kodu działający gdy już jest jakieś gui do dokomentowania gdy bedzie gui zmienie to najwyżej jak juz gui bedzie działało jakos
    """
    def load_data_from_desktop(self):
       
        # Jeśli GUI działa, filedialog sam znajdzie "rodzica".

        file_path = filedialog.askopenfilename(title="Wybierz obraz hiperspektralny", filetypes=[("Pliki NPZ", "*.npz")])

        if not file_path:
            return #nothing chosen

        x = self.dm.load_image_from_path(file_path)
        if x is not None: #contains sth
            self.data = x
    """

    # fragment kodu działający gdy nie ma naszego okienka do zakomentowania na potem
    def load_data_from_desktop(self): #choosing one file form users desktop that program will process

        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)  #pojawi się jako pierwsza karta dodatkowo

        # Otwórz dialog
        file_path = filedialog.askopenfilename(title="Choose hiperspektral image", filetypes=[("Pliki NPZ", "*.npz")])

        root.destroy()

        if not file_path:
            return #nothing chosen

        x = self.dm.load_image_from_path(file_path)
        if x is not None: #contains sth
            self.data = x

    @abstractmethod
    def show_gui(self):
        #everyone different dashboard
        pass


    @abstractmethod
    def predict_average(self):
        pass #rózne dla userów naukowiec będzie mógł jeszcze podejrzeć score np















#na chwile obecną rolnik tylko będzie mógł tylko dodawać obraz hiperspektralny oraz rozpoczynać processing z podglądem wyników
class Farmer(User):
    def __init__(self, config: Config, user_id: int):
        super().__init__(config, user_id)
        self.model_manager = ModelManager(config)

    def show_gui(self):
        #TODO what should farmer see - implement her sth form gui
        pass






    def predict_average(self):
        if self.data is None:
            #here add popup siglnal from view that there is no data
            return None
        if self.model_manager.model_type is None:
            #here add popup signal that model in not loaded
            return None




        #miejsce na zmianę obrazu na taki na którym można dokonać predykcji:
        processed_spectrum = None #zmienna ostateczna na której dokonamy predykcji





        result = self.model_manager.predict(processed_spectrum)
        return result








#rowniez moze załadować załadować i sprawdzic wyniki -  w tym score ale ma również dostęp dostęp do modeli oraz folderu z danymi testowymi, trenignowymi do przetrenowania oraz
#moze ekslorować wynilo
class Scientist(User):

    def __init__(self, config: Config, user_id: int):
        super().__init__(config, user_id)
        # also can see diagrams of data
        self.explorer = Explorer()
        self.model_manager = ModelManager(config)

    def show_gui(self):
        # TODO what should sceintist see - implement her sth form gui
        pass

    def predict_average(self):
        if self.data is None:
            #here add popup siglnal from view that there is no data
            return None
        if self.model_manager.model_type is None:
            #here add popup signal that model in not loaded
            return None




        #miejsce na zmianę obrazu na taki na którym można dokonać predykcji:
        processed_spectrum = None #zmienna ostateczna na której dokonamy predykcji





        result = self.model_manager.predict(processed_spectrum)
        return result


    #Wartosci 0 to np nasz las, 1 to baseline
    def choosing_model(self, chosen_model: int):


        if(chosen_model == 0):
            self.predict_average()
        else:
            # TODO place for baseline idk loading datata making model prediction ....
            pass















