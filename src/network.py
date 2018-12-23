from vector      import vector
from layer       import layer
from collections import defaultdict
from pathlib     import Path

import networkm
import networkf

#global 
TESTING_SET = networkm.data_set()




'''
to do:
-back propagation
-fix up messing code 
-clean up input sequence 
'''

def main():
    run_count = 0

    Input = get_data(run_count)[0]
    hidden_1 = layer(16, "h") #where the 'magic' happens
    hidden_1.init_weights(len(Input))

    hidden_2 = layer(16, "h") # where the 'magic' happens
    hidden_2.init_weights(len(hidden_1))

    output = layer(10, "o")
    output.init_weights(len(hidden_2))

    layers = [hidden_1, hidden_2, output]

    if Path("../data/weights.txt").exists():
        layers = networkm.load_data("weights")

    mode = input('b/f')
    if mode == "b":
        back_propagation(layers)
    if mode == "f":
        forward_propagation(layers, run_count)



def forward_propagation(L: list, run_count: int, r=False):
    start = 'y'

    if start == 'y':
        run = True
    correct = 0
    while run or run_count != len(TESTING_SET):
        Input = get_data(run_count)[0] # layer with the 768 pixel values

        hidden_1_activation = layer_prop(Input, L[0])
        hidden_2_activation = layer_prop(hidden_1_activation, L[1])
        outputs = layer_prop(hidden_2_activation, L[2]).get_vector()


        guess = network_guess(outputs, [0,1,2,3,4,5,6,7,8,9])
        actual = get_data(run_count)[1]

        if guess == actual:
            correct += 1
        
        print("Guess: {}   Actual: {}\n".format(guess, actual))


        if r == True:
            return L, outputs
        print(outputs)
        start = input("Continue? ")

        if start == "n":
            run = False 

        run_count += 1


def back_propagation(L: "list of layers"):
    run = 0
    gradient = defaultdict(lambda: defaultdict(dict))

    learning_rate = int(input("learning rate: "))

    correct = 0
    while run != 5000 :
        L, outputs = forward_propagation(L, run, True)

        actual = get_data(run)[1]
        guess = network_guess(outputs, [0,1,2,3,4,5,6,7,8,9])

        if actual == guess:
            correct += 1

        print("Correct: {}".format(correct))

        dcost_gradient = change_in_cost(run, outputs)
        print("run: {}".format(run))


        for layer_i in range(len(L) - 1, -1, -1):
            nodec = 0 
            for node in L[layer_i]:
                if L[layer_i].t == 'o': #last layer back propogation
                    dcost = dcost_gradient[nodec]
                    dweight = delta_weight(node.get_sum(), L[layer_i - 1].activations.get_vector(), dcost)
                    dbias   = delta_bias(node.get_sum(), dcost)

                elif layer_i == 0: #first layer back propogation
                    dcost = delta_cost(L[layer_i + 1], gradient["L"+ str(layer_i + 1)]["gradient"], node)
                    dcost_gradient.append(dcost)
                    dweight = delta_weight(node.get_sum(), get_data(run)[0], dcost)
                    dbias   = delta_bias(node.get_sum(), dcost)

                else:               #hidden layers
                    dcost = delta_cost(L[layer_i + 1], gradient["L"+ str(layer_i + 1)]["gradient"], node)
                    dcost_gradient.append(dcost)
                    dweight = delta_weight(node.get_sum(), L[layer_i - 1].activations.get_vector(), dcost)
                    dbias   = delta_bias(node.get_sum(), dcost)

                #record the gradient
                #changes are recorded in the gradient for every change
                gradient["L"+str(layer_i)][nodec]["dcost"] = dcost
                gradient["L"+str(layer_i)][nodec]["dweights"] = dweight
                gradient["L"+str(layer_i)][nodec]["dbias"] = dbias


                #changes in node
                node.set_bias(node.get_bias() - dbias)
                change_node_w(dweight,node,learning_rate)
                nodec += 1

            gradient["L"+str(layer_i)]["gradient"] = vector(dcost_gradient)
        print("ERROR:{}\n".format(gradient["L"+str(layer_i)]["gradient"].sum()))
        run += 1
        dcost_gradient = []
    print("Correct Rate: {}".format(correct/run))
    networkm.save_data(L, "weights")


def change_node_w(delta: list, node: "node object", learning_rate=1):
    r_weights = []
    weights = node.get_weights()
    if len(weights) != len(delta):
        raise ValueError("Different lengths")

    for i in range(len(delta)):
        r_weights.append(weights[i] - (learning_rate*delta[i]))

    node.set_weights(vector(r_weights))

def delta_cost(L: "layer object", d_N_gradient: vector, N:"node object"):
    '''
    sigmoid_prime(z) * summation of W mutiplied by cost
    '''
    sum_w_delta = []
    for node in range(len(L)):
        sum_w_delta.append(sum(L[node].get_weights() * d_N_gradient[node]))
        
    
    return networkf.sigmoid_prime(N.get_sum()) * sum(sum_w_delta)

def delta_weight(node_sum: float , L_p_activation:list, dcost:float):
    delta = []
    
    for i in range(len(L_p_activation)):
        delta.append(networkf.sigmoid_prime(node_sum) * L_p_activation[i] * dcost)
    return delta

def delta_bias(node_sum: float, dcost: float):
    return networkf.sigmoid_prime(node_sum) * dcost


    
def change_in_cost(run:int, O:list):
    expected = []
    delta = []
    for l in range(len(O)):
        expected.append(0)

    actual = get_data(run)[1]
    expected[actual-1] = 1

    for expect_i in range(len(expected)):
        delta.append(networkf.cost_prime(O[expect_i], expected[expect_i]))
    return delta

    

def get_data(run: int):
    pixels, expected= TESTING_SET[run]
    pixels = convert_grey(pixels)
    Input = vector(pixels)
    return (Input,expected)




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
