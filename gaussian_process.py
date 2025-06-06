import numpy as np
from scipy.special import kv, gamma
import pandas as pd

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """Compute the RBF kernel between x1 and x2"""
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T
    d = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    return variance * np.exp((-0.5 / length_scale**2) * d)

def matern_kernel(x1, x2, nu=1.5, length_scale=1.0):
    """Compute the Matern kernel between two vectors x and y."""
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T
    d = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    d = np.where(d == 0, 1, d) # replace 0 distances with 1

    factor = np.sqrt(2 * nu) * d / length_scale
    coef = (2 ** (1. - nu)) / gamma(nu)
    return coef * (factor ** nu) * kv(nu, factor)

def periodic_kernel(x1, x2, length_scale=1.0, period=24.0, variance=1.0):
    x1 = np.atleast_2d(x1).T
    x2 = np.atleast_2d(x2).T
    dists = np.pi * np.abs(x1 - x2.T) / period
    return variance * np.exp(-2 * (np.sin(dists)**2) / length_scale**2)


def gp_posterior(X_train, y_train, X_test, kernel, noise=1e-8, **kwargs):
    """Compute posterior mean and covariance of GP"""
    n = len(X_train)
    n_tst = len(X_test)

    K = kernel(X_train, X_train, **kwargs) + noise * np.eye(n) # kernel fcn between train data 
    b = kernel(X_train, X_test, **kwargs) # kernel fcn between train and test data
    c = kernel(X_test, X_test, **kwargs) + noise * np.eye(n_tst) # kernel fcn between test data
    n = X_train.shape[0]

    K_inv = np.linalg.inv(K)


    mu_s = b.T.dot(K_inv).dot(y_train) # mean for test points
    cov_s = c - b.T.dot(K_inv).dot(b) # cov for test points

    #mu_t = K.T.dot(K_inv).dot(y_train)
    #cov_t = K - K.T.dot(K_inv).dot(K)

    #log_marg_like = -0.5*y_train.T.dot(K_inv).dot(y_train) - 0.5*np.log(np.abs(K)) - (n/2)*np.log(2*np.pi)

    return mu_s, cov_s#, mu_t, cov_t

def rmse(y_true, y_hat):
	if(y_true.shape != y_hat.shape):
		print("y_hat and y_true must be the same shape")
		assert(False)

	return np.sqrt(np.sum((y_hat - y_true)**2)/len(y_true))


if __name__ == '__main__':
    df = pd.read_csv("Metro_Interstate_Traffic_Volume.csv")
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df['year'] = df['date_time'].dt.year
    df['time_step'] = (df['date_time'] - pd.Timestamp("1970-01-01")) // pd.Timedelta(hours=1)

    df_train = df[df['date_time'].dt.year == 2017]
    df_test = df[(df['date_time'].dt.year == 2018) & (df['month']==1)]

    # Test points
    X_train = df_train['time_step']
    X_test = df_test['time_step']

    y_train = df_train['traffic_volume']
    y_test = df_test['traffic_volume']

    print(rbf_kernel(X_train, X_train).shape)
    print(periodic_kernel(X_train, X_train).shape)
    print(matern_kernel(X_train, X_train).shape)