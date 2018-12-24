
import gzip
import struct
from pathlib    import Path    
from PIL        import Image
from layer      import layer
from vector     import vector

#globals
DATA_TYPES = {
    8: ('ubyte', 'B', 1),
    9: ('byte   ', 'b', 1),
    11: ('>i2', 'h', 2),
    12: ('>i4', 'i', 4),
    13: ('>f4', 'f', 4),
    15: ('>f8', 'd', 8)
    }

def label_data():
    '''
    loads label data from MNIST file
    '''

    print("Training set labels")
    f = open("../data/train-labels.idx1-ubyte",'rb')
    f.seek(0)
    magic_num = struct.unpack('>4B',f.read(4))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][0]))
    print("data format: {}".format(DATA_TYPES[magic_num[2]][1]))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][2]))

    print()
    f.seek(4)
    num_labels = struct.unpack('>I',f.read(4))[0]
    print("labels:   {}".format(num_labels))
    print()
    labels = []
    for i in range(num_labels):
        labels.append(struct.unpack(">B",f.read(1))[0])
    f.close()
    return labels
    

def image_data():
    '''
    reads image data from the MNIST data set 
    '''

    print("Training set images")
    f = open("../data/train-images.idx3-ubyte",'rb')
    f.seek(0)
    magic_num = struct.unpack('>4B',f.read(4))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][0]))
    print("data format: {}".format(DATA_TYPES[magic_num[2]][1]))
    print("data type: {}".format(DATA_TYPES[magic_num[2]][2]))

    print()
    f.seek(4)
    num_images = struct.unpack('>I',f.read(4))[0]
    rows = struct.unpack('>I',f.read(4))[0]
    columns = struct.unpack('>I',f.read(4))[0]
    print("images: {}".format(num_images))
    print("rows:   {}".format(rows))
    print("columns:{}".format(columns))
    print()
    images = []
    for i in range(num_images):
        images.append(struct.unpack('>' + 'B'*(rows*columns)\
                                    ,f.read(rows*columns)))
    f.close()
    return images


def data_set():
    '''
    zips the set of image pixel data 
    along with the set of label test data 
    '''
    images = image_data()
    labels = label_data()
    return [i for i in zip(images, labels)]

def save_image(pixels:list):
    '''
    saves the image given a list of pixels 
    '''
    image = Image.new('L', (28,28))
    image.putdata(pixels)
    image.save("test.png")


def save_data(L: list, name: str):
    '''
    saves the weights and bias values of a given list of layers objects
    into a .txt file
    '''
    save_f = open("../data/{}".format(name) + ".txt", 'w')
    for l_index in range(len(L)):
        save_f.write("L{},{}\n".format(L[l_index].t, len(L[l_index])))

        for node_index in range(len(L[l_index])):
            save_f.write("{},{}:{}/{}\n".format(l_index, node_index, \
                                        L[l_index][node_index].get_weights().get_vector(), \
                                        L[l_index][node_index].get_bias()))
        save_f.write("\n")
    save_f.close()


def load_data(name: str) -> []:
    '''
    reads a saved .txt file containing weights and biases 
    and translates into a list of layer objects
    '''
    L = []
    save_f = Path("../data/"+name+".txt")

    if save_f.exists():
        f = open(save_f, "r")
        for line in f:
            if line[0] == "L":
                line = line.rstrip()
                line = line.split(",")
                L.append(layer(int(line[1]), line[0][1]))
            elif line[0] != "\n":
                if L != []:
                    index = line.strip().split(":")[0]
                    data = line.strip().split(":")[1]

                    index = index.split(",")
                    L[int(index[0])][int(index[1])].\
                                    set_weights(vector(eval(data.split("/")[0])))
                    L[int(index[0])][int(index[1])].set_bias(float(data.split("/")[1])) 
            else:
                pass    
        f.close()           
    else:
        raise AttributeError("save file does not exist")
    return L



##to do
'''
- create memoize decorator
'''
