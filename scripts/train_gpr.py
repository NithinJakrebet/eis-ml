from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def train_gpr(X_train, y_train, length_scale=1.0, noise_level=0.1, n_restarts=10):
    """ Train a Gaussian Process Regressor on given data. """
    
    kernel = RBF(length_scale=length_scale) + WhiteKernel(noise_level=noise_level)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=n_restarts)
    
    gp.fit(X_train, y_train)
    return gp  # Return trained model
