##Version 2 March 20

from scipy.stats import ortho_group  # ortho_group.rvs random orthogonal matrix
import numpy as np                   # np.random.rand  random matrix
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import pylab 
from sklearn.cluster import KMeans
## ADMM for SSC
class ssc_model(object):
    def __init__(self, X, affine=False, alpha1=800,  alpha2 = None, thr=0.0002, maxIter=200):
        self.alpha1 = alpha1 
        if not alpha2:
            self.alpha2 = alpha1
        else:
            self.alpha2 = alpha2
        
        self.X = X
        self.affine = affine    
        self.thr = thr
        self.maxIter = maxIter
        self.N = X.shape[1]   # number of samples
        
        self.T = np.dot(self.X.T,self.X)
        T1 = np.abs(self.T - np.diag(np.diag(self.T)))
        self.lambda1 = np.min(np.max(T1,axis=1))
        self.mu1 = self.alpha1/self.lambda1
        self.mu2 = self.alpha2 
        self.I = np.eye(self.N,dtype=np.float32)
        self.ones = np.ones((self.N,self.N),dtype=np.float32)
        self.vec1N = np.ones((1,self.N),dtype = np.float32)
        self.err =[]
        self.errList = []
    def computeCmat(self):
        if not self.affine:
            A = np.linalg.inv(self.mu1*self.T + self.mu2*self.I)
            C1 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda2 = np.zeros((self.N,self.N),dtype=np.float32)
            err = 10*self.thr
            iter1 = 1
            while (err>self.thr)and(iter1<self.maxIter):
                #update Z
                Z = np.dot(A,self.mu1*self.T + self.mu2*(C1 - Lambda2/self.mu2))
                Z = Z - np.diag(np.diag(Z))
                # update C
                tmp_val = np.abs(Z + Lambda2/self.mu2) - (self.ones/self.mu2)
                C2 = np.maximum(0,tmp_val)*np.sign(Z + Lambda2/self.mu2)
                C2 = C2 - np.diag(np.diag(C2))
                # update lagrangian multipliers
                Lambda2 = Lambda2 + self.mu2*(Z-C2)
                # compute errors
                tmp_val = np.abs(Z - C2)
                err = np.max(tmp_val.reshape(-1,1))
                self.errList = np.append(self.errList, err)
                C1 = C2
                iter1 = iter1 +1
                # print('the error is = %f' % err)
        else:
            A = np.linalg.inv(self.mu1*self.T + self.mu2*self.I+ self.mu2*self.ones)
            C1 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda2 = np.zeros((self.N,self.N),dtype=np.float32)
            Lambda3 = np.zeros((1,self.N),dtype=np.float32)
            err1 = 10*self.thr
            err3 = 10*self.thr
            iter1 = 1
            while ((err1>self.thr)or(err3>self.thr))and(iter1<self.maxIter):
                #update Z
                tmp_val = self.mu1*self.T + self.mu2*(C1-Lambda2/self.mu2) + self.mu2*np.dot(self.vec1N.T,(self.vec1N - Lambda3/self.mu2))
                Z = np.dot(A,tmp_val)
                Z = Z - np.diag(np.diag(Z))
                # update C
                tmp_val = np.abs(Z + Lambda2/self.mu2) - (self.ones/self.mu2)
                C2 = np.maximum(0,tmp_val)*np.sign(Z + Lambda2/self.mu2)
                C2 = C2 - np.diag(np.diag(C2))
                # update lagrangian multipliers
                Lambda2 = Lambda2 + self.mu2*(Z-C2)
                Lambda3 = Lambda3 + self.mu2*(np.dot(self.vec1N,Z) - self.vec1N)
                # compute errors
                tmp_val = np.abs(Z - C2)
                err1 = np.max(tmp_val.reshape(-1,1))
                tmp_val = np.abs(np.dot(self.vec1N,Z) - self.vec1N)
                err3 = np.max(tmp_val.reshape(-1,1))
                
                C1 = C2
                iter1 = iter1 + 1
                print('iter1 = %d, the error 1 is = %f and error 2 is %f' % (iter1, err1, err3))
        return C2


def make_pd_line_in_rn(p, D, amount=1000):
   # D is the dimension we draw our subspaces from
   # p is the dimension of the subspace we want to draw (eg p=2 => line, p=3 => plane, etc)
   # assume that D >= p
   coeffs = ortho_group.rvs(D)[:p]
   t = np.random.rand(amount, p) - 0.5
   return np.matmul(t, coeffs), coeffs


t = 100 #Number of data points
n = 3 #Number of linear subspaces
Nl = 10 #Number of points from each subspace is set to Nl and are assumbed to be equal in this setup
D = 3 #Dimension of subspace
clusters = 3
N = Nl*clusters #Total number of data points in Y
dl = D #Dimention of each subspace is set to be equal as in D
p = 2

Y = np.zeros([D,N])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for cl in range(clusters):
    X, coeffsX = make_pd_line_in_rn(p,D, t)
# Now randomly sample data points after sampling from Xis.
# Need not sample and can generate Y diretly from function "make_pd_line_in_rn".
    ax.scatter(X[:,0],X[:,1],X[:,2])#,color = 'Orange')
    temp = np.random.randint(t, size = Nl)
    Y[:, Nl*cl:Nl*(cl+1)] = np.transpose(X[temp,:]) 
# Check for rank of Y (sanity check). it should be same as dl in this case. 
print(linalg.matrix_rank(Y))

# For now we are setting permutation matrix to be Identity matrix. For ease of implementation.
G = np.eye(N)
Y = np.matmul(Y,G) 

SSC_Model = ssc_model(Y, False, 1000)
C = SSC_Model.computeCmat()

# print(np.diag(C))
C = C / np.linalg.norm(C, ord = np.inf, axis = 0)

W = np.abs(C) + np.abs(np.transpose(C))
print(np.shape(W))
# Construct diagonal matrix D
DiagMatrix = np.diag(np.sum(W, axis=1))
D_ = np.linalg.inv(np.sqrt(DiagMatrix))
L = np.dot(np.dot(D_, W), D_)
w, v = np.linalg.eigh(L)
k = np.argmin(np.ediff1d(np.flipud(w))) + 1
print('k = %d.' % k)
# Extract k largest eigenvectors
XFinal = v[:, N - k:]

# Construct matrix Y by renormalizing X
YFinal = np.divide(XFinal, np.reshape(np.linalg.norm(XFinal, axis=1), (XFinal.shape[0], 1)))
# Cluster rows of Y into k clusters using K-means 
kmeans = KMeans(n_clusters=k, random_state=1234).fit(YFinal)
cluster_labels = kmeans.labels_
print(cluster_labels)
ax.scatter(Y[0,:],Y[1,:],Y[2,:],c = cluster_labels,marker = '^')#,color = 'Orange')

plt.show()