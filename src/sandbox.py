#!/usr/bin/python3
# -*- coding: utf-8 -*-
import sys
import traceback

import mnist_loader

def main():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    return 0

if __name__ == "__main__":
    r = main()
    sys.exit(r)
