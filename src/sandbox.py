#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import traceback

import mnist_loader
import network

import jsonpickle # pip install jsonpickle
import json
import yaml # pip install pyyaml

# Third-party libraries
import numpy as np


def main():
    
    # training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    # net = network.Network([784, 30, 10])
    # net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    net = network.Network([4, 3, 2])
    serialized = jsonpickle.encode(net)
    #print(json.dumps(json.loads(serialized), indent=4))

    #print("===================\n")

    sizes = [4,2,1]
    print( sizes[1:] )
    print("===================\n")

    a = np.random.randn(2, 1) 
    print( a )
    
    print("===================\n") 
    a = np.random.randn(1, 1) 
    print( a )

    print("======== biases ===========\n") 
    biases = [np.random.randn(y, 1) for y in sizes[1:]];
    print( biases ) 


    print("======== weights ===========\n")  

    #weights = [np.random.randn(y, x)  for x, y in zip(sizes[:-1], sizes[1:])]
    #  [4,2] , [2:1]  =>  [(4,2), (2,1)]


    #a = zip(sizes[:-1], sizes[1:])
    #print( a ) 

    #serialized = jsonpickle.encode(a)
    #print(json.dumps(json.loads(serialized), indent=4))

    weights = [np.random.randn(y, x)  for x, y in zip(sizes[:-1], sizes[1:])]
    print( weights  ) 
    
    print("===================\n") 
    a = zip(biases, weights) 
    num = 0
    for b,w in a:
        print("  = %d = \n" % num)
        print(b)
        print("---\n")
        print(w)

        num = num + 1 


    return 0

if __name__ == "__main__":
    r = main()
    sys.exit(r)
