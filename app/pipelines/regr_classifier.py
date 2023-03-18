import pickle


class LRClassifier():
    def __init__(self):
        self.df_path = '../datasets/heart_cleveland_upload.csv'
        with open("../models/model.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, data):
        result = self.model.predict(data)
        return result
