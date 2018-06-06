import network

def or_gate(a, b):
    # define data
    data_or = [[0, 0, 0],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 1]]

    net = network.Network(data_or)
    return net.predict(a, b)

def and_gate(a, b):
    # define data
    data_and = [[0, 0, 0],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 1]]
    
    net = network.Network(data_and)
    return net.predict(a, b)

def nor_gate(a, b):
    # define data
    data_nor = [[0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        [1, 1, 0]]
    
    net = network.Network(data_nor)
    return net.predict(a, b)

def nand_gate(a, b):
    # define data
    data_nand = [[0, 0, 1],
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]]
    
    net = network.Network(data_nand)
    return net.predict(a, b)
