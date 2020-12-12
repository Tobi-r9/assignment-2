import numpy as np
import matplotlib.pyplot as plt


def generate_data(mu, tau, N):
    data = np.random.normal(mu, 1/tau,N)
    return data

def init_E_t(a,b):
    return a/b

def set_aN(a,N):
    return a+N/2

def comp_mean(data):
    return np.mean(data)

def set_muN(lmbda, mu, N ,x_mean):
    temp = (lmbda*mu+N*x_mean)/(lmbda+N)
    return N

def update_lmbda(lmbda, N, E_t):
    return (lmbda+N) * E_t

def update_bN(lmbda,mu,b, data, E_mu, E_mu2):
    temp1 = sum(data**2 - 2*data*E_mu+E_mu2)
    temp2 = lmbda*(E_mu2-2*mu*E_mu + mu**2)
    return b +0.5*(temp1+temp2)

def moment2_mu(data, N, E_t):
    return comp_mean(data)**2 +1/(N*E_t)

def update_Et(a_n,b_n):
    return a_n/b_n[-1]

def q_mu(E_t, lmbda_0, mu_0, mu, data):
    temp1 = 0
    for d in data:
        temp1 += (d-mu)**2
    #temp1 = sum(data-mu)**2
    temp2 = lmbda_0*(mu-mu_0)**2
    return -E_t/2 * (temp1+temp2)

def q_tau(tau,N,a_0, b_n,b_0): 
    return (a_0-1)*np.log(tau)-b_0*tau+N/2 * np.log(tau) - (b_n-b_0)*tau 

def estimate_posterior(mu, tau, E_t, lmbda_0, mu_0, data, b_n, b_0, N, a_0):
    temp1 = q_mu(E_t, lmbda_0, mu_0, mu, data)
    temp2 = q_tau(tau,N,a_0, b_n,b_0)
    return np.exp(temp1+temp2)

def real_posterior(mu, tau, N, data, mean, b_0, a_0,lmbda_0, mu_0):
    avg = 0
    for d in data:
        avg += (d-mean)**2

    temp1 = tau**(N/2+a_0-0.5)*np.exp(-tau*(0.5*avg+b_0))
    temp2 = np.exp(-tau*0.5*(lmbda_0*(mu-mu_0)**2+N*(mean-mu)**2))
    return temp1*temp2

def plot_posteriors(E_t, lmbda_0, mu_0, data, b_n,b_0, N, a_0, mean,tau,mu):
    x = np.linspace(mu-0.5,mu+0.5,100)
    y = np.linspace(tau-0.5,tau+0.5,100)
    X,Y = np.meshgrid(x,y)
    p = estimate_posterior(X, Y, E_t, lmbda_0, mu_0, data, b_n,b_0, N, a_0)
    plt.contour(X,Y,p,colors='red',label='approximation')
    z = real_posterior(X, Y, N, data, mean, b_0, a_0,lmbda_0,mu_0)
    plt.contour(X,Y,z,colors='blue',label='real') 
    plt.title(r'$a_0 = {}, b_0 = {}, \mu_0 = {}, \lambda_0 = {}$'.format(a_0,b_0,mu_0,lmbda_0))
    plt.xlabel(r'$\mu$')
    plt.ylabel(r'$\tau$')
    plt.show()

def main():

    #initialize variables
    np.random.seed(123)
    N = 100
    a_0 = 1
    b_0 = 1
    mu_0 = 0
    lmbda_0 = 1
    tau = np.random.gamma(a_0, 1/b_0)
    mu = np.random.normal(mu_0, np.sqrt((lmbda_0 * tau))**-1)
    data = generate_data(mu, tau, N)
    E_mu = comp_mean(data)
    a_n = set_aN(a_0,N)
    E_t = a_0/b_0
    lmbda_n = [lmbda_0]
    b_n = [b_0]
    mean = comp_mean(data)

    for _ in range(3):
        plot_posteriors(E_t, lmbda_0, mu_0, data, b_n[-1] ,b_0, N, a_0, mean,tau,mu)
        E_mu2 = moment2_mu(data, N, E_t)
        b_n.append(update_bN(lmbda_0,mu,b_0, data, E_mu, E_mu2))
        E_t = update_Et(a_n,b_n)
        lmbda_n.append(update_lmbda(lmbda_0, N, E_t))
        

    plt.figure(figsize=(8, 3))
    plt.subplot(121)
    plt.plot(b_n, label='b_N')
    plt.ylabel("b_n")
    plt.xlabel("Iterations")
    plt.subplot(122)
    plt.plot(lmbda_n, label=r'$\lambda$')
    plt.ylabel(r"$\lambda$")
    plt.xlabel("Iterations")
    plt.legend(loc=(1.04, 0))
    plt.show()


if __name__ == "__main__":
    main()