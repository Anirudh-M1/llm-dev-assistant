def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n - 1)

def add(d, c):
    return d + c

def is_prime(n):
    if n < 2:
        return False
    for divisor in range(2, int(n ** 0.5) + 1):
        if n % divisor == 0:
            return False
    return True

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a