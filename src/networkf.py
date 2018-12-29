#global
E = 2.718281828459045

class Memoize:
    def __init__(self,f):
        self.f = f
        self.cache = {}

    def __call__(self,*args):
        if args in self.cache:
            return self.cache[args]
        else:
            answer = self.f(*args)        # Recursive calls will set cache too
            
            self.cache[args] = answer
            return answer

    def reset_cache(self):
        self.cache = {}

@Memoize
def sigmoid(x):
    "the sigmoid function"
    return 1.0/(1.0+(E**(-x)))

@Memoize
def sigmoid_prime(x):
    "derivative of the sigmoid function"
    "growth of the function"
    return sigmoid(x)*(1 - sigmoid(x))

@Memoize
def cost(out, target):
    return (0.5)*((target - out)**2)

@Memoize
def cost_prime(out, target):
    return -(target - out)

