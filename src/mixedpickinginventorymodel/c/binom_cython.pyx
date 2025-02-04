import math
def binom_cython(unsigned int k, unsigned int n, double p):
    return math.factorial(n)/(math.factorial(k)*math.factorial(n-k))*(p**k)*((1-p)**(n-k))