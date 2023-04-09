import numpy as np
from scipy.linalg import norm



class fixed_point_schemes():
    def __init__(self, x_0, f, m=5, K_max=1E3, tol=1E-5, theta=1E-2, tau=1E-3, D=1E6, eps=1E-6, alpha=.1, method='aa1_prs'):
        self.f = f
        self.g = lambda x: x-f(x)
        x_0 = np.asarray(x_0)
        self.n = x_0.size
        n = self.n
        self.m = m
        m = self.m
        self.theta = theta
        self.tau = tau
        self.alpha = alpha
        # H = Id + sum_i v_iw_i^T
        self.vs = np.zeros((m,n))
        self.ws = np.zeros_like(self.vs)

        self.x_km1 = x_0
        self.x_k = f(x_0)
        self.xtilde_km1 = self.x_km1
        self.xtilde_k = self.x_k
        self.m_k = 0
        self.n_AA = 0
        g_km1 = self.g(self.x_km1)
        self.norm_g_0 = norm(g_km1)
        self.DU = D*self.norm_g_0
        self.shat_ks = np.zeros_like(self.vs)
        self.eps = eps
        self.tol = tol
        self.K_max = int(K_max)
        if method == 'aa1-safe':
            self.step = self.step_aa1prs
        elif method == 'original':
            self.step = self.step_original
        elif method == 'aa1':
            self.step = self.step_aa1
        else:
            raise Exception('Invalid solver given.')
        

    def run(self):
        norm_g_ks = [1]
        for k in range(self.K_max):
            x_k, norm_g_k = self.step()
            norm_g_k = norm_g_k/self.norm_g_0
            norm_g_ks.append(norm_g_k)
            if norm_g_k < self.tol or norm_g_k > 1/self.tol:
                break
        else:
            print('Did not converge.')
        return norm_g_ks
    
    def step_aa1(self):
        return self.step_aa1prs(*(3*[False]))

    def step_aa1prs(self, powell=True, restart=True, safeguard=True):
        self.m_k += 1
        s_km1 = self.xtilde_k-self.x_km1
        g_km1 = self.g(self.x_km1)
        y_km1 = self.g(self.xtilde_k)-g_km1
        self.shat_ks[:-1,:] = self.shat_ks[1:,:]
        self.shat_ks[-1,:] = np.zeros(self.n)
        a = self.shat_ks@s_km1
        b = np.linalg.norm(self.shat_ks, axis=1)**2
        self.shat_ks[-1,:] = s_km1-self.shat_ks.transpose()@(np.divide(a,b,out=np.zeros_like(a), where=b!=0))
        
        # Restart checking
        if (self.m_k == self.m+1) or (norm(self.shat_ks[-1,:])<self.tau*norm(s_km1)) and restart:
            self.m_k = 0
            self.shat_ks = np.zeros((self.m, self.n))
            self.shat_ks[-1,:] = s_km1
            self.vs = np.zeros_like(self.shat_ks)
            self.ws = np.zeros_like(self.vs)
        
        # Powell regularisation
        if powell:
            gamma_km1 = self.shat_ks[-1,:].transpose()@self.eval_H(y_km1)/(norm(self.shat_ks[-1,:])**2)
            theta_km1 = self.phi(gamma_km1, self.theta)
            ytilde_km1 = theta_km1*y_km1-(1-theta_km1)*g_km1
        else:
            ytilde_km1 = y_km1

        # Update H
        Hytilde_km1 = self.eval_H(ytilde_km1)
        self.vs[:-1,:] = self.vs[1:,:]
        self.vs[-1,:] = s_km1-Hytilde_km1
        self.ws[:-1,:] = self.ws[1:,:]
        self.ws[-1,:] = 0
        self.ws[-1,:] = self.eval_Ht(self.shat_ks[-1,:])/(self.shat_ks[-1,:].transpose()@Hytilde_km1)
        
        self.xtilde_km1 = self.xtilde_k
        g_k = self.g(self.x_k)
        self.xtilde_k = self.x_k-self.eval_H(g_k)
        self.x_km1 = self.x_k
        norm_g_k = norm(g_k)

        # Safeguarding
        if norm_g_k<=self.DU*(self.n_AA+1)**(-(1+self.eps)) or not safeguard:
            self.x_k = self.xtilde_k
            self.n_AA += 1
        else:
            self.x_k = self.alpha * self.x_k + (1-self.alpha)*self.f(self.x_k)
        return self.x_k, norm_g_k
    
    def eval_H(self, x):
        return x + self.vs.transpose()@(self.ws@x)
    
    def eval_Ht(self, x):
        return x + self.ws.transpose()@(self.vs@x)
    

    def step_original(self):
        self.x_k = self.f(self.x_k)
        norm_g_k = norm(self.g(self.x_k))
        return self.x_k, norm_g_k


    def phi(self, x, theta):
        if np.abs(x)>= theta:
            return 1
        elif x <0:
            return (1+theta)/(1-x)
        else:
            return (1-theta)/(1-x)