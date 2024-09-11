import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class Perceptron:
    def __init__(self,data,random_state=-1):
        self.features = []
        for entry in data:
            self.features.append(entry["data"])
        self.actuals = []
        for entry in data:
            self.actuals.append(entry["class"])

        self.converged = False
        self.w = np.zeros(len(self.features[0]))
        self.b = 0
        self.max_pass = 10
        self.mistake = 1
        if not random_state == -1: 
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.actuals,random_state=random_state, test_size=0.2,stratify=self.actuals)
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.features, self.actuals, test_size=0.2,stratify=self.actuals)
    
    def predict(self):
        y_pred = []
        for entry in self.X_test:
            y_pred.append(np.sign(np.dot(entry, self.w) + self.b))

        return accuracy_score(self.y_test, y_pred)
        # print(confusion_matrix(self.y_test, y_pred))
        # print(accuracy_score(self.y_test, y_pred))

    def run(self,iterations=10000):
        for i in range(iterations):
            for index, selected_element in enumerate(self.X_train):
                dot_product = np.dot(selected_element,self.w)
                if self.y_train[index] * np.sign((dot_product + self.b)) <= 0:
                    self.w += (self.y_train[index] * selected_element)
                    self.b += self.y_train[index]
                    self.mistake += 1
        
        # print(self.w)
