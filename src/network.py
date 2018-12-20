from array import array
from layer import layer

import networkm


def main():
    Input = networkm.single_image() # layer with the 768 pixel values
    hidden_1 = layer(16, "h") #the layers
    hideen_2 = layer(16, "h") # where the 'magic' happens
    output = [0,1,2,3,4,5,6,7,8,9]
    print(len(Input))










if __name__ == "__main__":
    main()