import numpy as np
import matplotlib.pyplot as plt


def hypothesis(theta, x):
    return np.dot(x, theta)


def cost(mytheta, X, y, m):
    """Cost Function"""
    return float((1./(2*m)) * np.dot((hypothesis(mytheta, X)-y).T, (hypothesis(mytheta, X)-y)))


def gradient_descent(X, Y, m, alpha, iterations, theta_start=np.zeros(2)):
    """Gradient Descent"""

    theta = theta_start
    jvec = []  # Used to plot cost as function of iteration
    thetahistory = []  # Used to visualize the minimization path later on
    for meaninglessvariable in range(iterations):
        tmptheta = theta
        jvec.append(cost(theta, X, Y, m))
        thetahistory.append(list(theta[:, 0]))

        # Simultaneously updating theta values
        for j in range(len(tmptheta)):
            tmptheta[j] = theta[j] - \
                (alpha/m)*np.sum((hypothesis(theta, X) - Y)
                                 * np.array(X[:, j]).reshape(m, 1))
        theta = tmptheta
    return theta, thetahistory, jvec
