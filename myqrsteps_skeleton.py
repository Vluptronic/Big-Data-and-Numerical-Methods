import numpy as np

##Code snippets you'll need in no particular order

#right-hand sides
#1) A[np.ix_(i,j)]-2*v@(v.T@A[np.ix_(i,j)])
#2) A
#3) sigma+v[0]
#4) np.eye(m,m)

#extended snippet 
#1)  A[k,k] = sigma
#    A[k+1:m,k] = 0

#indices
#1) range(k,m)
#2) range(k+1,n)

def myqrsteps(A):

    m,n = A.shape

    #initialize Q
    #-------- your code begins here: fill the right hand side-------
    Q = np.eye(m, m)
    #-------- your code ends here: fill the right hand side-------

    # Householder reduction to triangular form.
    for k in range(0,min(m-1,n)):
        # Introduce zeros below the diagonal in the k-th column.
        # Use Householder transformation, I - rho*u*u'.
        
        # Define appropriate row indices
        #-------- your code begins here--
        i = range(k, m)
        #-------- your code ends here ---
   
        u = A[i,k][:,np.newaxis]
        sigma = np.linalg.norm(u)

        #Define v for reflection matrix
        v = -u
        #-------- your code begins here: fill the right hand side-------
        v[0] = sigma + v[0]
        
        #-------- your code begins here: fill the right hand side-------
        v = v/np.linalg.norm(v) #normalization
   
        # Skip transformation if column is already zero.
        if sigma != 0:

            # Update the k-th column.
            # --- your codes begin here---
            A[k, k] = sigma
            A[k + 1:m, k] = 0
            # --- your codes end here -----
    
    
            # Apply the transformation to remaining columns of A.
      
            # Define appropriate column indices
            #-------- your code begins here--
            j = range(k + 1, n)
            #-------- your code ends here--

            #Apply the reflection here
            #-------- your code begins here: fill the right hand side-------
            A[np.ix_(i, j)] = A[np.ix_(i, j)] - 2 * v @ (v.T @ A[np.ix_(i, j)])
            #-------- your code ends here: fill the right hand side-------
      
            #Update Q matrix
            Q[:,i] = Q[:,i]-2*Q[:,i]@(v@v.T)

    #Define upper triangular matrix
    #-------- your code begins here: fill the right hand side-------
    R = A 
    #-------- your code ends here: fill the right hand side-------

    return Q, R
