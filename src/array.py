#array class
#for vectors (matricies with n x 1 column)
class array:
    def __init__(self, array=[]: list):
        if type(array) != list:
            raise AttributeError("Must be type array")
        self.array = array 

    def __repr__(self):
        return "array({})".format(self.array)

    def __mul__(self, right):
        if type(right) in [float, int]:
            result = self.array.copy()
            for index in range(len(self.array)):
                result[index] *= right
            return result
        elif type(right) == array and len(right.array) == len(self.array):
            result = 0
            for index in range(len(self.array)):
                result += (right.array[index] * self.array[index])
            return result
        else:
            raise TypeError("Mutiplication Error")

    def __rmul__(self,left):
        return self * left

    def __add__(self, right):
        if type(right) == array and len(self.array) == len(right.array):
            result = []
            for e_index in len(range(self.array)):
                result[e_index] = self.array[e_index] + right.array[e_index]
            return result
        else:
            raise TypeError("Cannot be added")

    def __len__(self):
        return len(self.array)           

    def get_array(self):
        return self.array

    def add(self, arg):
        self.array.append(arg)
    




