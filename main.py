from perceptron import Perceptron
from utilities import Utilities

utilities = Utilities()

utilities.parse_input("./data/diabetes_scale.txt")

perceptron = Perceptron(utilities.data)
perceptron.run()
perceptron.predict()
