from vector import vector
from layer import layer

import networkm

#global 
TESTING_SET = networkm.data_set()


def main():
    run_count = 0

    Input = input_layer(run_count)
    hidden_1 = layer(16, "h") #where the 'magic' happens
    hidden_1.init_weights(len(Input))

    hidden_2 = layer(16, "h") # where the 'magic' happens
    hidden_2.init_weights(len(hidden_1))

    output = layer(10, "o")
    output.init_weights(len(hidden_2))

    layers = [hidden_1, hidden_2, output]
    forward_propagation(layers, run_count)



def forward_propagation(L: list, run_count: int):
    if "file" in "directory":
        pass # read files function 
    else:
        print()
        start = input("Start program (y/n) ")

        if start == 'y':
            run = True

        while run or run_count != len(TESTING_SET):
            Input = input_layer(run_count) # layer with the 768 pixel values

            hidden_1_activation = layer_prop(Input, L[0])
            hidden_2_activation = layer_prop(hidden_1_activation, L[1])
            outputs = layer_prop(hidden_2_activation, L[2])


            guess = network_guess(outputs, [0,1,2,3,4,5,6,7,8,9])
            print()
            print("Guess: {}".format(guess))
            print("Actual: {}".format(TESTING_SET[run_count][1]))

            start = input("Continue? ")
            save = input("Save? ")

            if save == "y":
                networkm.save_data(L, 'weights')
                Layers = networkm.load_data("weights")
                print(Layers[0][0].get_weights())


            if start == "n":
                run = False 

            run_count += 1

def input_layer(run: int):
    pixels, expected= TESTING_SET[run]
    pixels = convert_grey(pixels)
    Input = vector(pixels)
    return Input


def layer_prop(A: vector,L: layer):
    '''
    takes layer object and returns the activations of it
    '''
    L.set_inputs(A)
    L.calculate()

    return L.activations


def network_guess(O: [], possible: []):
    if len(O) != len(possible):
        raise ValueError("Output and possible results are different length")
    guess = 0
    for network_guess in range(len(O)):
        if O[network_guess] >=  O[guess]:
            guess = network_guess
    return possible[guess]

def convert_grey(pixels: [] or ()):
    return [i for i in map(lambda x: x/255, [p for p in pixels])]







    







if __name__ == "__main__":
    main()  
