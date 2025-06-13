import numpy as np
from scipy.special import kv, gamma # needed for matern kernel implementation
import pandas as pd

def rbf_kernel(x1, x2, length_scale=1.0, variance=1.0):
    """
    Compute the RBF kernel between x1 and x2. For p-many features, m1 and m2 many examples in x1 and x2, we have
    
    x1 = m1 x p
    x2 = m2 x p 
    """
    x1 = np.atleast_2d(x1) # ensure proper dimensions
    x2 = np.atleast_2d(x2)

    d = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    return variance * np.exp((-0.5 / length_scale**2) * d)

def matern_kernel(x1, x2, nu=1.5, length_scale=1.0, variance = 1.0):
    """Compute the Matern kernel between two vectors x and y."""
    x1 = np.atleast_2d(x1) # ensure proper dimensions
    x2 = np.atleast_2d(x2)

    d = np.sqrt(np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T))  
    d = np.where(d == 0, 1e-8, d) # avoid divide by 0

    factor = np.sqrt(2 * nu) * d / length_scale
    coef = (2 ** (1. - nu)) / gamma(nu)
    return variance * coef * (factor ** nu) * kv(nu, factor)

def periodic_kernel(x1, x2, length_scale=1.0, period=24.0, variance=1.0):
    x1 = np.atleast_2d(x1)
    x2 = np.atleast_2d(x2)

    #d = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)   
    #d = np.linalg.norm(x1[:, None, :] - x2[None, :, :], axis=2) # get distances
    

    # Compute Gram matrix
    K = np.zeros((x1.shape[0], x2.shape[0]))
    for d in range(x1.shape[1]): # iterate over each feature
        x1_d = x1[:, d][:, None]
        x2_d = x2[:, d][None, :]

        d = np.abs(x1_d - x2_d) # absolute dist
        sin_term = np.sin(np.pi * d / period) ** 2
        K += -2 * sin_term / length_scale**2 
    return variance*np.exp(K)


def negative_log_marginal_likelihood(X, y, kernel, noise=1e-8, K_lst = None, **kwargs):
    """Compute log marginal likelihood.
    K_lst = [K, K^-1] contains the kernel matrix and already computed inverse if desired. Otherwise will compute
    """
    if K_lst is None:
        print('Computing K')
        K = kernel(X, X, **kwargs) + noise * np.eye(len(X))
        print("Inverting K")
        K_inv = np.linalg.inv(K)
    else:
        K, K_inv = K_lst[0], K_lst[1] 
    print("Inversion Done")
    
    alpha = np.dot(K_inv, y)
    sign, logdet = np.linalg.slogdet(K) # compute log determinant

    log_likelihood = -0.5 * np.dot(y.T, alpha) - 0.5 * logdet -0.5 * len(X) * np.log(2 * np.pi)

    return -log_likelihood

def gp_posterior(X_train, y_train, X_test, kernel, noise=1e-8, get_cov = False, **kwargs):
    """Compute posterior mean and covariance of GP"""
    n = len(X_train)
    n_tst = len(X_test)

    print("Computing Gram Matrix")
    K = kernel(X_train, X_train, **kwargs)
    K_noisy = K + noise * np.eye(n) # kernel fcn between train data (plus noise)
    b = kernel(X_train, X_test, **kwargs) # kernel fcn between train and test data
    c = kernel(X_test, X_test, **kwargs)# kernel fcn between test data

    print('Inverting Gram Matrix')
    K_inv = np.linalg.inv(K_noisy) # inverting noisy gram matrix

    print('Computing Mean and Covariance')
    mu_s = b.T.dot(K_inv.dot(y_train)) # mean for test points
    mu_tr = K.T.dot(K_inv.dot(y_train))

    if get_cov == True:
        cov_s = c - (b.T).dot((K_inv).dot(b)) # cov for test points
        cov_tr = K - (K.T).dot(K_inv.dot(K))
    else:
        cov_s, cov_tr = None, None
    return mu_s, cov_s, mu_tr, cov_tr, K, K_inv

def mse(y_true, y_hat):
    '''Computes Mean Squared Error between y_true and y_hat'''
    return np.sum((y_hat - y_true)**2)/len(y_true)

def rmse(y_true, y_hat):
    '''Computes Root Mean Squared Error between y_true and y_hat'''
    return np.sqrt(np.sum((y_hat - y_true)**2)/len(y_true))

def mae(y_true, y_hat):
    '''Computes Mean Absolute Error between y_true and y_hat'''
    return np.sum(np.abs(y_hat - y_true))/len(y_true)


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