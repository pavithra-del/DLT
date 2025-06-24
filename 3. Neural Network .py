import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

X, y = load_breast_cancer(return_X_y=True)
y = y.reshape(-1, 1)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

W1 = np.random.randn(X.shape[1], 10)
b1 = np.zeros((1, 10))
W2 = np.random.randn(10, 1)
b2 = np.zeros((1, 1))

def sigmoid(z): return 1 / (1 + np.exp(-z))
def tanh_deriv(a): return 1 - a**2

losses = []
for epoch in range(1000):
    Z1 = X_train @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2)
    loss = -np.mean(y_train*np.log(A2+1e-9)+(1-y_train)*np.log(1-A2+1e-9))
    losses.append(loss)
    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2
    db2 = dZ2.sum(0, keepdims=True)
    dZ1 = (dZ2 @ W2.T) * tanh_deriv(A1)
    dW1 = X_train.T @ dZ1
    db1 = dZ1.sum(0, keepdims=True)
    W1 -= 0.01 * dW1
    b1 -= 0.01 * db1
    W2 -= 0.01 * dW2
    b2 -= 0.01 * db2
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

A2_test = sigmoid(np.tanh(X_test @ W1 + b1) @ W2 + b2)
print("Accuracy: %.2f%%" % (np.mean((A2_test > 0.5) == y_test) * 100))

plt.plot(losses)
plt.title("Loss")
plt.show()
