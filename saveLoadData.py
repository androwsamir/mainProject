import pickle


class SaveLoadData:

    def saveModel(self, dataToSave, fileName):
        with open(fileName, 'wb') as file:
            pickle.dump(dataToSave,file)

    def loadModel(self, fileName):
        with open(fileName, 'rb') as file:
            np = pickle.load(file)

        return np