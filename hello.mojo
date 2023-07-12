# Simple example to demonstrate Mojo, 
# Range and print functions
from Range import range
from Object import object
from IO import print

fn main():
    print("hello mojo")
    for x in range(9, 0, -3):
        print(x)