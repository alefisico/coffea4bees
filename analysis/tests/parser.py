import sys
import argparse
import unittest

#https://stackoverflow.com/questions/35270177/passing-arguments-for-argparse-with-unittest-discover

class UnitTestParser(object):

    def __init__(self):
        self.args = None

    def parse_args(self):
        # Parse optional extra arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-i','--inputFile',     help='file to be tested')
        parser.add_argument('-c','--knownCounts',   help='yaml file with known counts')
        parser.add_argument('-e','--error_threshold',   default=0.001, help='yaml file with known counts')
        ns, args = parser.parse_known_args()
        self.args = vars(ns)

        # Now set the sys.argv to the unittest_args (leaving sys.argv[0] alone)
        sys.argv[1:] = args

wrapper = UnitTestParser()
