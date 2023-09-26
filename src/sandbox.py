#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import traceback

import mnist_loader
import network

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = network.Network([784, 30, 10])

    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)

    return 0

if __name__ == "__main__":
    r = main()
    sys.exit(r)
