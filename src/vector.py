#vector class
#for vectors (matricies with n x 1 column)

class vector(list):
    def __init__(self, vector):
        if type(vector) == tuple:
            vector = list(vector)
        if type(vector) != list:
            raise AttributeError("Must be type list")
        self.vector = vector

    def __repr__(self):
        return "vector({})".format(self.vector)
    
    def __str__(self):
        return "vector({})".format(self.vector)
        
    def __mul__(self, right):
        """overloads the multiplication class 
        the output is the dot product of the two vectors"""
        if type(right) in [float, int]:
            result = self.vector.copy()
            for index in range(len(self.vector)):
                result[index] *= right
            return result
        elif type(right) == vector and len(right.vector) == len(self.vector):
            result = 0
            for index in range(len(self.vector)):
                result += (right.vector[index] * self.vector[index])
            return result
        else:
            raise TypeError("Mutiplication Error")

    def __rmul__(self,left):
        return self * left

    def __add__(self, right):
        if type(right) == vector and len(self.vector) == len(right.vector):
            result = []
            for e_index in len(range(self.vector)):
                result[e_index] = self.vector[e_index] + right.vector[e_index]
            return result
        else:
            raise TypeError("Cannot be added")
    
    def __setitem__(self, index, value):
        if index < len(self.vector) - 1 and type(index) == int:
            self.vector[index] = value
        else:
            if type(index) != int:
                raise TypeError("index is type {}".format(type(index)))
            else:
                raise IndexError("index out of range")

    def __getitem__(self, index):
        if type(index) != int:
            raise TypeError("index must be of type int")
        if index > len(self.vector):
            raise IndexError
        else:
            return self.vector[index]


    def __len__(self):
        return len(self.vector)           

    def get_vector(self):
        return self.vector

    def add(self, arg):
        self.vector.append(arg)
    
    def clear(self):
        self.vector = []


    




