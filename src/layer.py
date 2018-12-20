# layer class for 
import random
from node import node


class layer:
    def __init__(self, nodes=0, activate =[], t='h'):
        '''creates a layer class object based on the number of nodes
            then initializs the number of nodes specified
            
            "t"(type) specifies the type of layer
            s = starting layer
            h = hidden layer 
            o = output layer
            
        '''
        if t not in ['s','h','o']:
            raise AttributeError("layer type does not exist")
        
        self.node_amt = nodes
        self.nodes = []
        self.t = t
        self.create_nodes()

        
    def create_nodes(self):
        node_count = 0 
        while node_count != self.node_amt:
            self.nodes.append(node())

    
    def init_weights(self, N):
        '''
        N is the number of nodes in the previous layers
        '''
        for node in self.nodes:
            node.init_weights(N)

    def __setitem__(self, index, value):
        '''
        setting the weight of the node at that index
        '''
        if index < len(self.nodes) or index > len(self.nodes) - 1:
            raise IndexError
        
        self.nodes[index].w.set_weight(index, value)

    def __getitem__(self, index):
        return self.nodes[index]

    def __iter__(self):
        for node in self.nodes:
            yield node 
    
    def __len__(self):
        return len(self.nodes)

