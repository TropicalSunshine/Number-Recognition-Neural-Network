#global
E = 2.718281828459045

@Memoize
def sigmoid(x):
    "the sigmoid function"
    if x < (-700):
        return 0
    return 1.0/(1.0+(E**(-x)))

@Memoize
def sigmoid_prime(x):
    "derivative of the sigmoid function"
    "growth of the function"
    return sigmoid(x)*(1 - sigmoid(x))

@Memoize
def cost(actual, expected):
    return (actual - expected)**2

@Memoize
def cost_prime(actual, expected):
    return 2*(actual - expected)

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
