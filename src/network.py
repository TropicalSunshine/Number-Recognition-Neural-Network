from vector import vector
from layer import layer

import networkm


def main():
    # for testing program
    Input = vector(networkm.single_image()) # layer with the 768 pixel values
    hidden_1 = layer(16, "h") #where the 'magic' happens
    hidden_1.init_weights(len(Input))

    hidden_2 = layer(16, "h") # where the 'magic' happens
    hidden_2.init_weights(len(hidden_1))

    output = layer(10, "o")
    output.init_weights(len(hidden_2))


    hidden_1_activation = forward_prop(Input, hidden_1)
    hidden_2_activation = forward_prop(hidden_1_activation, hidden_2)
    outputs = forward_prop(hidden_2_activation, output)

    print(outputs)
    guess = 0
    for network_guess in range(len(outputs)):
        if outputs[network_guess] >=  outputs[guess]:
            guess = network_guess
    print("Guess: {}".format(guess))






def forward_prop(A: vector,L: layer):
    '''
    takes layer object and returns the activations of it
    '''
    L.set_inputs(A)
    L.calculate()

    return L.activations




    







if __name__ == "__main__":
    main()  
