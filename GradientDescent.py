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
            print(updatedError, prevError)
            converged = True
        prevError = updatedError
        if(iteration == max_iter):
            converged = True
        iteration += 1
    return p0, p1


# if __name__ == '__main__':

#     x, y = make_regression(n_samples=100, n_features=1, n_informative=1,
#                         random_state=0, noise=35)
#     print 'x.shape = %s y.shape = %s' %(x.shape, y.shape)

#     alpha = 0.01 # learning rate
#     ep = 0.01 # convergence criteria

#     # call gredient decent, and get intercept(=theta0) and slope(=theta1)
#     theta0, theta1 = gradient_descent(alpha, x, y, ep, max_iter=1000)
#     print ('theta0 = %s theta1 = %s') %(theta0, theta1)

#     # check with scipy linear regression
#     slope, intercept, r_value, p_value, slope_std_error = stats.linregress(x[:,0], y)
#     print ('intercept = %s slope = %s') %(intercept, slope)

#     # plot
#     for i in range(x.shape[0]):
#         y_predict = theta0 + theta1*x

#     pylab.plot(x,y,'o')
#     pylab.plot(x,y_predict,'k-')
#     pylab.show()
#     print "Done!"