import numpy as np

class Utilities:
    def __init__(self):
        self.data = []
        
    def parse_input(self,file_name):
        with open(file_name) as file:
            for line in file:
                split_line = line.rstrip().split(" ")
                observation = {}
                parameters = []
                for entry in split_line:
                    if entry == '+1' or entry == '-1':
                        if entry == "+1":
                            observation["class"] = 1
                        if entry == "-1":
                            observation["class"] = -1
                    else:
                        split_feature = entry.split(":")
                        parameters.append(float(split_feature[1]))
                    
                    observation["data"] = np.array(parameters)

                if len(observation["data"]) == 8:
                    self.data.append(observation)