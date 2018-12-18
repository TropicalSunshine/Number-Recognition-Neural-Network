# layer class for 
import random
import node


class layer:
    def __init__(self, nodes=0):
        """creates a layer class object based on the number of nodes"""
        self.node_amt = nodes
        self.nodes = []
        
        create_nodes()

    def create_nodes():
        node_count = 0 
        while node_count != self.node_amt:
            
    def __len__(self):
        return len(self.nodes)