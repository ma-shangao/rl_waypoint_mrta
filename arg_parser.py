import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-N', type=int, default=50, help="Number of task points")
parser.add_argument('-k', type=int, default=3, help="Number of clusters")
parser.add_argument('-F', type=int, default=2, help="dimension of task point feature")
