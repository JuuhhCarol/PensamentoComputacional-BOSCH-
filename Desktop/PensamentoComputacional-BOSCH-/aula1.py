import math as m
import random as rd

def S(x, w, b):
    return (x * w) + b
#calculo feito antes da funcao de ativacao

def sigmoid(S):
    return 1 / (1 + m.exp(-S))
#funcao de ativacao sigmoide

t = 1.00
#target = valor "desejado"

x = 1.6
#valor
w = rd.random()
#weight/peso
b = rd.random()
#bias/vies

lr = 0.2
#learning rate/taxa de aprendizado
epochs = 1000
#"quantas vezes ele aprende"

for i in range(epochs):
    s = S(x, w, b)
    #o calculo antes da funcao de ativacao
    y = sigmoid(s)
    #a funcao de ativacao aplicada na variavel anterior

    dEdW = (y - t) * y * (1 - y) * x #derivada do erro pelo peso
    dEdb = (y - t) * y * (1 - y) #derivada do erro pelo vies

    w = w - lr * dEdW #peso atualizado
    b = b - lr * dEdb #vies atualizado

    print(sigmoid(S(x, w, b)))
    #com os valores de peso e vies atualizados, aplica o calculo
    #antes da funcao de ativacao, e executa a funcao de ativacao sigmoide
    #e printa isso
    pass