from playdoh import *

def fun(x):
    return x**2

if __name__ == '__main__':
    print map(fun, [1,2], cpu=2)

