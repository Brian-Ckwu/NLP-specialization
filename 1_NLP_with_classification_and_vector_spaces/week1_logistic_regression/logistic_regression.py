import numpy as np
from tqdm import tqdm

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def gradient_descent(x, y, theta, alpha, num_iters):
    m = x.shape[0]
    for _ in tqdm(range(0, num_iters)):
        # get z, the dot product of x and theta
        z = x @ theta
        # get the sigmoid of z
        s = np.vectorize(sigmoid)
        h = s(z)
        # calculate the cost function
        J = (-1 / m) * ((y.T @ np.log(h)).item() + ((1 - y.T) @ np.log(1 - h)).item())
        # update the weights theta
        theta = theta - alpha / m * (x.T @ (h - y))

    J = float(J)
    return J, theta

if __name__ == "__main__":
    # Check the function
    # Construct a synthetic test case using numpy PRNG functions
    np.random.seed(1)
    # X input is 10 x 3 with ones for the bias terms
    tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
    # Y Labels are 10 x 1
    tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

    # Apply gradient descent
    tmp_J, tmp_theta = gradient_descent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
    print(f"The cost after training is {tmp_J:.8f}.")
    print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")