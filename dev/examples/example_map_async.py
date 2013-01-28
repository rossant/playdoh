from playdoh import *

def fun(x):
    return x**2

if __name__ == '__main__':
    task = map_async(fun, [1,2], cpu=2)
    print task.get_result()

