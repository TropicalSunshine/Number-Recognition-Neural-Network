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
-fix up messy code 
-clean up input sequence 
'''

def main():
    run_count = 0

    # network layout for a MNIST training set images 
    # Input layer - takes in the 784 grey scale pixels (range 0 - 1) each node contains the value of a single pixel
    # Hidden layer 1 - contains 16 nodes 784 weights per node
    # Hidden layer 2 - contains 16 nodes 16 weights per node
    # Output layer   - contains 10 nodes and 16 weights per node
    # largest interval of the output layer's activations corresponds to the network's guess of what that number is


    #-----------------------change here----------------------------
    Input = get_data(run_count)[0]
    hidden_1 = layer(16, "h")
    hidden_1.init_weights(len(Input))

    hidden_2 = layer(16, "h") 
    hidden_2.init_weights(len(hidden_1))

    output = layer(10, "o")
    output.init_weights(len(hidden_2))

    layers = [hidden_1, hidden_2, output]
    #--------------------------------------------------------------


    #checks for saved weights and bias file
    if Path("../data/weights.txt").exists():
        layers = networkm.load_data("weights")

    print("f) forward prop")
    print("b) backward prop")
    print("s) sketch")
    mode = input()

    if mode == "b":
        back_propagation(layers)
    if mode == "f":
        forward_propagation(layers, run_count)



def forward_propagation(L: list, run_count: int, r=False):
    '''
    forward propagation 
    first layer activations is taken as inputs for the next layer nodes
    weight vector node length corresponds to the amount of nodes in the previous layer
    after the dot product of the activation vector of the previous layer along with current node weight vector and inputted into the 
    sigmoid function 
    then the bias is added 
    repeats until the last layer is reached
    '''
    start = 'y'

    if start == 'y':
        run = True
    correct = 0
    while run or run_count != len(TESTING_SET):
        Input = get_data(run_count)[0] # layer with the 768 pixel values


        #---------------change here---------------------------
        hidden_1_activation = layer_prop(Input, L[0])
        hidden_2_activation = layer_prop(hidden_1_activation, L[1])
        outputs = layer_prop(hidden_2_activation, L[2]).get_vector()
        #-----------------------------------------------------

        #calculates the guess of the network 
        guess = network_guess(outputs, [0,1,2,3,4,5,6,7,8,9])
        actual = get_data(run_count)[1]

        if guess == actual:
            correct += 1
        
        
        print("Guess: {}   Actual: {}\n".format(guess, actual))
        
        
        #returns the final outputs
        #returns list of layers
        if r == True:
            return outputs
        
        display(run_count)

        start = input("Continue? ")

        if start == "n":
            run = False 

        run_count += 1


def back_propagation(L: "list of layers"):
    '''
    Using gradient descent methods 
    back propagation is asssited through find the change in weights of a single node 
    and the change in bias of that node 
    calculated through the chain rule 
    back propagation begins with the last layer
    and works its way back until it reaches the first layer
    '''

    # interval for place in data to begin training
    totalError = 0.0
    run = 0
    begin = run 

    gradient = defaultdict(lambda: defaultdict(dict))

    #learning rate mutiplier 
    learning_rate = float(input("learning rate: "))

    correct = 0
    
    while run !=  100: #sets the place in the data set to stop training
        outputs = forward_propagation(L, run, True)
        actual = get_data(run)[1]

        guess = network_guess(outputs, [0,1,2,3,4,5,6,7,8,9])

        if actual == guess:
            correct += 1

        print("Correct: {}".format(correct))
        dcost_gradient = delta_cost_o(run, outputs)
        print("run: {}".format(run))
        error = total_error(outputs, run)
        print("error: {}".format(error))

        #summing up the errors for later calculation of total error 
        totalError += error


        for layer_i in range(len(L) - 1, -1, -1):
            nodec = 0 
            for node in L[layer_i]:
                if L[layer_i].t == 'o': #output layer back propogation
                    dcost = dcost_gradient[nodec]
                    dweight = delta_weight(dcost, node.get_sum(), L[layer_i - 1].activations.get_vector() )
                    dbias   = delta_bias(dcost, node.get_sum())
                    #print("outputs: {}\n weights:{} \n dcost: {}\n dweight:{}\n dbias: {}\n p_activations: {}\n bias:{}\n z:{}\n".format(outputs, node.get_weights(), dcost, dweight, dbias, L[layer_i - 1].activations.get_vector(), node.get_bias(), node.get_sum()))

                elif layer_i == 0: #first layer back propogation
                    dcost = delta_cost(L[layer_i + 1], gradient["L"+ str(layer_i + 1)]["gradient"], nodec)
                    dcost_gradient.append(dcost)
                    dweight = delta_weight(dcost, node.get_sum(), get_data(run)[0])
                    dbias   = delta_bias(dcost, node.get_sum())

                else:               #hidden layers
                    dcost = delta_cost(L[layer_i + 1], gradient["L"+ str(layer_i + 1)]["gradient"], nodec)
                    dcost_gradient.append(dcost)
                    dweight = delta_weight(dcost, node.get_sum(), L[layer_i - 1].activations.get_vector())
                    dbias   = delta_bias(dcost, node.get_sum())

                #record the gradient
                #changes are recorded in the gradient for every change
                gradient["L"+str(layer_i)][nodec]["dweights"] = dweight
                gradient["L"+str(layer_i)][nodec]["dbias"] = dbias


                #changes in node
                node_bias = node.get_bias()
                node.set_bias(node_bias - dbias)
                node.set_weights(change_node_w(dweight,node,learning_rate))

                nodec += 1
                
            gradient["L"+str(layer_i)]["gradient"] = vector(dcost_gradient)
            dcost_gradient = []

        run += 1

    print("Correct Rate: {}".format(correct/(run - begin)))
    print("total Error: {}".format(totalError/(run - begin)))
    networkm.save_data(L, "weights")



def display(run_count: int):
    image, label = TESTING_SET[run_count]
    networkm.display_image(image)


def change_node_w(delta: list, node: "node object", learning_rate=1):
    '''
    changes the weights in the node 
    corresponding to the delta index
    '''
    r_weights = []
    weights = node.get_weights()
    if len(weights) != len(delta):
        raise ValueError("Different lengths")

    for i in range(len(delta)):
        r_weights.append(weights[i] - (learning_rate*delta[i]))

    return vector(r_weights)

def delta_cost(L: "layer object", cost_gradient: vector, nodec: int):
    '''
    sigmoid_prime(z) * summation of W mutiplied by cost
    calculates the DE/da of the hidden layers
    '''

    if len(L) != len(cost_gradient):
        raise AttributeError

    L_dC_Da = []

    for node_i in range(len(L)):
        DE_da = cost_gradient[node_i]
        da_dznet = networkf.sigmoid_prime(L[node_i].get_sum())
        dznet_da = L[node_i].get_weights()[nodec]
        L_dC_Da.append(DE_da * da_dznet * dznet_da)
    return sum(L_dC_Da)


def delta_weight(dcost: float, znet: float, P_activation:list ):
    dweight = []
    da_dznet = networkf.sigmoid_prime(znet)
    for i in range(len(P_activation)):
        dznet_dw = P_activation[i]
        dweight.append(dcost * da_dznet * dznet_dw)
    return dweight



def delta_bias(dcost:float, node_sum:float):
    '''
    calculates the DE/db for the node
    '''
    return networkf.sigmoid_prime(node_sum) * dcost


def delta_cost_o(run:int, O:list):
    '''
    calculates the DE/da of the last layer (output layer)
    '''
    expected = []
    delta = []
    for l in range(len(O)):
        expected.append(0)

    actual = get_data(run)[1]
    expected[actual] = 1

    for expect_i in range(len(expected)):
        delta.append(networkf.cost_prime(O[expect_i], expected[expect_i]))
    return delta

    

def get_data(run: int):
    '''
    gets the MNIST database images 
    '''
    pixels, expected= TESTING_SET[run]
    pixels = convert_grey(pixels)
    Input = vector(pixels)
    return (Input,expected)


def total_error(O: list, run: int):
    expected = []
    for i in range(len(O)):
        expected.append(0)
    expected[get_data(run)[1]] = 1

    cost = []
    for i in range(len(O)):
        cost.append(networkf.cost(O[i], expected[i]))
    return sum(cost)

def layer_prop(A: vector,L: layer):
    '''
    takes layer object and returns the activations of it
    '''
    L.set_inputs(A)
    L.calculate()

    return L.activations


def network_guess(O: [], possible: []):
    '''
    calculates the neural networks guess of the given image 
    '''
    if len(O) != len(possible):
        raise ValueError("Output and possible results are different length")
    guess = 0
    for network_guess in range(len(O)):
        if O[network_guess] >=  O[guess]:
            guess = network_guess
    return possible[guess]

def convert_grey(pixels: [] or ()):
    '''
    converts image pixel data into 
    values from 0 to 1
    1 = 255
    0 = 0 
    '''
    return [i for i in map(lambda x: x/255, [p for p in pixels])]


if __name__ == "__main__":
    main()  
