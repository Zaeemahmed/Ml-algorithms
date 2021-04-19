import numpy as np

def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    m = x.shape[0]
    p0 , p1 = 0, 0
    prevError = sum((p0 + p1*x[i] - y[i]) ** 2 for i in range(m))
    converged = False
    iteration = 0
    while(not converged):
        cost0 = (1 / m) * sum(p0 + p1*x[i] - y[i] for i in range(m))
        cost1 = (1 / m) * sum((p0 + p1*x[i] - y[i])* x[i] for i in range(m))
        p0 += -alpha * cost0
        p1 += -alpha * cost1
        updatedError = sum((p0 + p1*x[i] - y[i]) ** 2 for i in range(m))
        if(abs(updatedError - prevError) < ep):
            converged = True
        prevError = updatedError
        if(iteration == max_iter):
            converged = True
        iteration += 1
    return p0, p1


