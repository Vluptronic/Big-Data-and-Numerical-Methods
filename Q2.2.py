import numpy as np

A = np.array([[1.,2.],[0.,1.],[1.,0.]])

# First calculate reduced Q,R

Q_hat, R_hat = np.linalg.qr(A,mode = 'reduced')
print("Q_reduced is", Q_hat)
print("R_reduced is", R_hat)


Q_complete, R_complete = np.linalg.qr(A, mode = "complete")
print("Complete_Q: ",Q_complete)
print("Complete_R: ",R_complete)

# Check Orthogonal

print("IdentityCheckQ_hat: ", Q_hat.T @ Q_hat)
print("IdentityCheckQ_complete: ", Q_complete.T@Q_complete)
