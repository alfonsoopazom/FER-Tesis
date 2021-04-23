import agente.policy as politica
import numpy as npm
from agente.policy import policyPrint, plus


def printNumpy():
    a = npm.arange(15).reshape(3, 5)
    print(a.shape)

def main():
    politica.policyPrint()
    policyPrint()
    plus(1,2)
    printNumpy()

if __name__ == '__main__':
    main()