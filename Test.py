from Scripts.MicroTensor import Mtensor
from Scripts import NeuralNet
from Scripts import Losses
from Scripts import Optimizers
from random import gauss, seed
import matplotlib.pyplot as plt

seed(1337)

PLOTLOSS = True
PLOTDATA = True
EPOCHS = 50
NB_POINTS = 10
SIGMA = 1

#Regression   
#X = [-5.0 + i * 0.5 for i in range(NB_POINTS)]
#y = [2 * x + 1 for x in X]

#Classification
#X = [
#[2.0, 3.0, -1.0], [3.0, -1.5, 1.0], [1.0, 0.0, 1.0], [1.0, 2.0, 1.0],
#[-4.0, -2.0, 1.0], [0.5, -0.5, 1.0], [5.0, -2.0, 1.0], [-1.0, 1.0, 0.0],
#[-1.0, 1.0, 0.5], [0.5, -3.0, -2.0]
#]
#y = [1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0]

model = NeuralNet.Sequential([
    NeuralNet.Linear(1, 1)
])

params = model.parameters()

Loss = Losses.MSELoss()
Optim = Optimizers.GradientDescent(params = params)
losses = list()

for _ in range(EPOCHS):
    yhat = model(X)
    lossi = Loss(yhat, y)
    losses.append(lossi.val)

    Optim.zero_grad()
    lossi.backward()
    Optim.step()

if PLOTLOSS:
    plt.figure()
    plt.plot(list(range(EPOCHS)), losses)
    plt.title("Loss function")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show(block=False)

yhat = [yhati.val for yhati in yhat]

if PLOTDATA:
    plt.figure()
    plt.plot(list(range(NB_POINTS)), y, c = 'b', label = "true")
    plt.plot(list(range(NB_POINTS)), yhat, c = 'r', label = "pred")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()