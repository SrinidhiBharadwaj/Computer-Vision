#!/usr/bin/env python
# coding: utf-8

# # CSE 252B: Computer Vision II, Winter 2023 – Assignment 2
# 
# ### Instructor: Ben Ochoa
# 
# ### Due: Wednesday, February 8, 2023, 11:59 PM

# ## Instructions
# * Review the academic integrity and collaboration policies on the course website.
# * This assignment must be completed individually.
# * This assignment contains both math and programming problems.
# * All solutions must be written in this notebook
# * Math must be done in Markdown/LaTeX.
# * You must show your work and describe your solution.
# * Programming aspects of this assignment must be completed using Python in this notebook.
# * Your code should be well written with sufficient comments to understand, but there is no need to write extra markdown to describe your solution if it is not explictly asked for.
# * This notebook contains skeleton code, which should not be modified (This is important for standardization to facilate efficient grading).
# * You may use python packages for basic linear algebra, but you may not use packages that directly solve the problem. If you are uncertain about using a specific package, then please ask the instructional staff whether or not it is allowable.
# * You must submit this notebook exported as a pdf. You must also submit this notebook as an .ipynb file.
# * Your code and results should remain inline in the pdf (Do not move your code to an appendix).
# * **You must submit 3 files on Gradescope - .pdf , .ipynb and .py file where the .py file is the conversion of your .ipynb to .py file . You must mark each problem on Gradescope in the pdf. You can convert you .ipynb to .py file using the following command:**
# 
# <center> jupyter nbconvert --to script filename.ipynb --output output_filename.py </center>
# 
# * It is highly recommended that you begin working on this assignment early.

# ## Problem 1 (Math): Line-plane intersection (5 points)
#   The line in 3D defined by the join of the points $\boldsymbol{X}_1 = (X_1,
#   Y_1, Z_1, T_1)^\top$ and $\boldsymbol{X}_2 = (X_2, Y_2, Z_2, T_2)^\top$
#   can be represented as a Plucker matrix $\boldsymbol{L} = \boldsymbol{X}_1
#   \boldsymbol{X}_2^\top - \boldsymbol{X}_2 \boldsymbol{X}_1^\top$ or pencil of points
#   $\boldsymbol{X}(\lambda) = \lambda \boldsymbol{X}_1 + (1 - \lambda) \boldsymbol{X}_2$
#   (i.e., $\boldsymbol{X}$ is a function of $\lambda$).  The line intersects
#   the plane $\boldsymbol{\pi} = (a, b, c, d)^\top$ at the point
#   $\boldsymbol{X}_{\boldsymbol{L}} = \boldsymbol{L} \boldsymbol{\pi}$ or
#   $\boldsymbol{X}(\lambda_{\boldsymbol{\pi}})$, where $\lambda_{\boldsymbol{\pi}}$ is
#   determined such that $\boldsymbol{X}(\lambda_{\boldsymbol{\pi}})^\top \boldsymbol{\pi} =
#   0$ (i.e., $\boldsymbol{X}(\lambda_{\boldsymbol{\pi}})$ is the point on
#   $\boldsymbol{\pi}$).  Show that $\boldsymbol{X}_{\boldsymbol{L}}$ is equal to
#   $\boldsymbol{X}(\lambda_{\boldsymbol{\pi}})$ up to scale.

# ### Solution:
# 
# To prove that $\boldsymbol{X}_{\boldsymbol{L}}$ is equal to $\boldsymbol{X}(\lambda_{\boldsymbol{\pi}})$ up to scale, we need to show that ${X}(\lambda_{{\pi}})$ = $\gamma$$X_{L}$, where $\gamma$ is a constant scale factor.
# 
# Consider, $\newline$$\newline$
#         &nbsp;&nbsp;&nbsp;X($\lambda_\pi)^T\pi$ = 0 $\newline$$\newline$
#         &nbsp;=>&nbsp;&nbsp;($\lambda_\pi X_1$ + (1 - $\lambda_\pi)X_2)^T$$\pi$ = 0 $\newline$$\newline$
#         &nbsp;=>&nbsp;&nbsp;$\lambda_\pi X_1^T$$\pi$ + $X_2^T\pi$ - $\lambda_\pi X_2^T\pi$ = 0 $\newline$$\newline$
#         &nbsp;=>&nbsp;&nbsp;$\lambda_\pi(X_1 - X_2)^T\pi$ = -$X_2^T\pi$ $\newline$$\newline$
#         &nbsp;=>&nbsp;&nbsp; $\lambda_\pi$ = $\frac{-1}{(X_1 - X_2)^T\pi}$$X_2^T\pi$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --> **1** $\newline$ $\newline$ $\newline$
# Substituting back the value of $\lambda_\pi$ in the pencil of points equation:
# 
# &nbsp;&nbsp;&nbsp;X($\lambda_\pi$) = $\lambda_\pi X_1$ + (1 - $\lambda_\pi) X_2$ $\newline$ $\newline$
# From **1**, $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp; $\frac{-1}{(X_1 - X_2)^T\pi}$$X_2^T\pi$$X_1$ + (1 - $\frac{-1}{(X_1 - X_2)^T\pi}$$X_2^T\pi$)$X_2$ $\newline$ $\newline$
# &nbsp;=>&nbsp;&nbsp; $\frac{X_2^T\pi}{(X_2 - X_1)^T\pi}$$X_1$ + $\frac{X_1^T\pi}{(X_2 - X_1)^T\pi}$$X_2$ $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp; $\frac{1}{(X_2 - X_1)^T\pi}$($X_1X_{2}^T$ - $X_2X_{1}^T$)$\pi$  $\newline$ $\newline$
# &nbsp;=>&nbsp;&nbsp; X($\lambda_\pi$) = $\frac{1}{(X_2 - X_1)^T\pi}$ L$\pi$ = $\frac{1}{(X_2 - X_1)^T\pi}$$X_L$ &nbsp;&nbsp;&nbsp;&nbsp; (replacing with pencil of points equation) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; --> **2**
# 
# 
# From 2, we can see that X($\lambda_\pi$) = $\gamma$$X_L$, where $\gamma$ is the scale factor which is equal to $\frac{1}{(X_2 - X_1)^T\pi}$.
# 
# <!--         
# Now, consider, $\newline$$\newline$
#         &nbsp;&nbsp;&nbsp;$X_L$ = L$\pi$
#         &nbsp;&nbsp;&nbsp; -->
#         
#        
# 
# 

# 

# ## Problem 2 (Math): Line-quadric intersection (5 points)
#   In general, a line in 3D intersects a quadric $\boldsymbol{Q}$ at zero, one
#   (if the line is tangent to the quadric), or two points.  If the
#   pencil of points $\boldsymbol{X}(\lambda) = \lambda \boldsymbol{X}_1 + (1 -
#   \lambda) \boldsymbol{X}_2$ represents a line in 3D, the (up to two) real
#   roots of the quadratic polynomial $c_2 \lambda_{\boldsymbol{Q}}^2 + c_1
#   \lambda_{\boldsymbol{Q}} + c_0 = 0$ are used to solve for the intersection
#   point(s) $\boldsymbol{X}(\lambda_{\boldsymbol{Q}})$.  Show that $c_2 =
#   \boldsymbol{X}_1^\top \boldsymbol{Q} \boldsymbol{X}_1 - 2 \boldsymbol{X}_1^\top \boldsymbol{Q}
#   \boldsymbol{X}_2 + \boldsymbol{X}_2^\top \boldsymbol{Q} \boldsymbol{X}_2$, $c_1 = 2 (
#   \boldsymbol{X}_1^\top \boldsymbol{Q} \boldsymbol{X}_2 - \boldsymbol{X}_2^\top \boldsymbol{Q}
#   \boldsymbol{X}_2 )$, and $c_0 = \boldsymbol{X}_2^\top \boldsymbol{Q} \boldsymbol{X}_2$.

# ### Solution:
# 
# General 3D intersection of a quadric **Q** can be written as, $\newline$
# 
# 
# $\boldsymbol{X}^T$__QX = 0__, where **X** representes a line. Using the pencil of points equation to represent a line, we can rewrite the equation as, $\newline$$\newline$
# &nbsp;&nbsp;&nbsp;$X(\lambda_Q)^T$QX($\lambda_Q$) = 0 $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp;($\lambda_Q X_1$ + (1 - $\lambda_Q$)$X_2)^T$Q($\lambda_Q X_1$ + (1 - $\lambda_Q)X_2$) = 0 $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp;($\lambda_Q X_1^T$ + $X_2^T$ - $\lambda_Q$$X_2^T$)Q($\lambda_Q X_1$ + $X_2$ - $\lambda_Q$$X_2$) = 0 $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp;($\lambda_Q X_1^TQ$ + $X_2^TQ$ - $\lambda_Q$$X_2^T$Q)($\lambda_Q X_1$ + $X_2$ - $\lambda_Q$$X_2$) = 0 $\newline$$\newline$
# &nbsp;=>&nbsp;&nbsp;$\lambda_Q^2 X_1^TQX_1$ + $\lambda_Q X_1^TQX_2$ - $\lambda_Q^2 X_1^TQX_2$ + $\lambda_Q X_2^TQX_1$ + $X_2^TQX_2$ - $\lambda_Q$$X_2^T$Q$X_2$ - $\lambda_Q^2 X_2^TQX_1$ - $\lambda_Q$$X_2^T$Q$X_2$ + $\lambda_Q^2 X_2^TQX_2$ = 0 $\newline$$\newline$
# 
# 

# Grouping $\lambda^2, \lambda$ and constant terms, <br/> <br/>
# 
# &nbsp;=>&nbsp;&nbsp; ($X_1^TQX_1$ - $X_1^TQX_2$ - $X_2^TQX_1$ + $X_2^TQX_2$)$\lambda_Q^2$ + 2($X_1^TQX_2$ - $X_2^T$Q$X_2$)$\lambda_Q$ + $X_2^TQX_2$ = 0
# 
# &nbsp;=>&nbsp;&nbsp; ($X_1^TQX_1$ - 2$X_1^TQX_2$ + $X_2^TQX_2$)$\lambda_Q^2$ + 2($X_1^TQX_2$ - $X_2^T$Q$X_2$)$\lambda_Q$ + $X_2^TQX_2$ = 0, &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; (since $X_1^TQX_2$ = $X_2^TQX_1$) &nbsp;&nbsp;&nbsp;&nbsp; --> **1**
# 
# From **1** we have, <br/><br/>
# 
# $c_2 =
#   \boldsymbol{X}_1^\top \boldsymbol{Q} \boldsymbol{X}_1 - 2 \boldsymbol{X}_1^\top \boldsymbol{Q}
#   \boldsymbol{X}_2 + \boldsymbol{X}_2^\top \boldsymbol{Q} \boldsymbol{X}_2$, $c_1 = 2 (
#   \boldsymbol{X}_1^\top \boldsymbol{Q} \boldsymbol{X}_2 - \boldsymbol{X}_2^\top \boldsymbol{Q}
#   \boldsymbol{X}_2 )$, and $c_0 = \boldsymbol{X}_2^\top \boldsymbol{Q} \boldsymbol{X}_2$.
# 

# 

# ## Problem 3 (Programming):  Linear Estimation of the Camera Projection Matrix (15 points)
#   Download input data from the course website.  The file
#   hw2_points3D.txt contains the coordinates of 50 scene points
#   in 3D (each line of the file gives the $\tilde{X}_i$, $\tilde{Y}_i$,
#   and $\tilde{Z}_i$ inhomogeneous coordinates of a point).  The file
#   hw2_points2D.txt contains the coordinates of the 50
#   corresponding image points in 2D (each line of the file gives the
#   $\tilde{x}_i$ and $\tilde{y}_i$ inhomogeneous coordinates of a
#   point).  The scene points have been randomly generated and projected
#   to image points under a camera projection matrix (i.e., $\boldsymbol{x}_i
#   = \boldsymbol{P} \boldsymbol{X}_i$), then noise has been added to the image point
#   coordinates.
# 
#   Estimate the camera projection matrix $\boldsymbol{P}_\text{DLT}$ using the
#   direct linear transformation (DLT) algorithm (with data
#   normalization).  You must express $\boldsymbol{x}_i = \boldsymbol{P} \boldsymbol{X}_i$
#   as $[\boldsymbol{x}_i]^\perp \boldsymbol{P} \boldsymbol{X}_i = \boldsymbol{0}$ (not
#   $\boldsymbol{x}_i \times \boldsymbol{P} \boldsymbol{X}_i = \boldsymbol{0}$), where
#   $[\boldsymbol{x}_i]^\perp \boldsymbol{x}_i = \boldsymbol{0}$, when forming the
#   solution. Return
#   $\boldsymbol{P}_\text{DLT}$, scaled such that
#   $||\boldsymbol{P}_\text{DLT}||_\text{Fro} = 1$
#   
#   The following helper functions may be useful in your DLT function implementation.
#   You are welcome to add any additional helper functions.

# In[1]:


#Additional Helper functions

def get_householder_matrix(x):
    dim, num_points = x.shape
    I = np.eye(dim)
    I = np.dstack([I]*num_points) #((3, 3, num_points))

    e1_vec = np.zeros(x.shape) #(1, 0, 0, 0)
    x_norm = np.linalg.norm(x, axis=0)
    e1_vec[0, :] = x_norm
    x1_sign = np.sign(x[0, :])
    v = x + (x1_sign * e1_vec)
    
    for i in range(num_points):
        v_i = v[:, i].reshape(-1, 1)
        I[:, :, i] = np.eye(dim) - 2 * (v_i @ v_i.T) / (v_i.T @ v_i)

    #Vectorization using einstein sum - Verified without vectorization as above
    #Uncomment the below line to run vectorized code (wrote out of curiosity of einsum)
    #I = I - 2 * (np.einsum("ji, li -> jli", v, v))/np.einsum("ji, ji->i", v, v)[None, None, :]
 
    return I[1:, :, :] #H_v[1:, :, :]

def get_kron_matrix(H_v, X):
    assert X.shape[1] == H_v.shape[2]
    A = np.zeros((2*H_v.shape[2], 12))
    j = 0
    for i in range(H_v.shape[2]):
        A[j:j+2, :] = np.kron(H_v[:, :, i], X[:, i].T)
        j+=2

    # Vectorization
    # H_v matrix is reshaped to 50x2x3x1 (from 2x3x3) 
    # X is reshaped to 50x1x1x4
    # Both matrices are multipled to produce the Kronecker output
    # Uncomment below two lines to run the vectorized form
#     A = np.transpose(H_v, (2, 0, 1))[:, :, :, None] * X.T[:, None, None, :]
#     A = A.reshape(100, 12)  
    
    return A

def get_p(A):
    _, _, vh = np.linalg.svd(A)
    return vh[-1, :]

def get_projection_matrix(x, X):
    H_v = get_householder_matrix(x)
    A = get_kron_matrix(H_v, X)
    P = get_p(A)
    return P.reshape(3, 4)


# In[2]:


import numpy as np
import time

def homogenize(x):
    # converts points from inhomogeneous to homogeneous coordinates
    return np.vstack((x,np.ones((1,x.shape[1]))))


def dehomogenize(x):
    # converts points from homogeneous to inhomogeneous coordinates
    return x[:-1]/x[-1]


def data_normalize(pts):
    # data normalization of n dimensional pts
    #
    # Input:
    #    pts - is in inhomogeneous coordinates
    # Outputs:
    #    pts - data normalized points
    #    T - corresponding transformation matrix
    
    """your code here"""
    
    T = np.eye(pts.shape[0]+1)
    
    assert pts.shape[0] in [2, 3]
    
    mean_x = np.mean(pts[0, :])
    mean_y = np.mean(pts[1, :])
    var_x = np.var(pts[0, :])
    var_y = np.var(pts[1, :])
    
    if pts.shape[0] == 2:
        s = np.sqrt(2/(var_x+var_y))
        T[0, 0], T[1, 1] = s, s
        T[0, 2] = -s*mean_x
        T[1, 2] = -s*mean_y
    else:
        mean_z = np.mean(pts[2, :])
        var_z = np.var(pts[2, :])
        s = np.sqrt(3/(var_x+var_y+var_z))
        T[0, 0], T[1, 1], T[2, 2] = s, s, s
        T[0, 3] = -s*mean_x
        T[1, 3] = -s*mean_y
        T[2, 3] = -s*mean_z
        
    pts = T @ homogenize(pts)
    return pts, T

def sum_of_square_projection_error(P, x, X):
    # Inputs:
    #    P - the camera projection matrix
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    # Output:
    #    cost - Sum of squares of the reprojection error
    
    """your code here"""
    
    x_homo = homogenize(X)
    x_est = P @ x_homo
    x_est = dehomogenize(x_est)    
    cost = x - x_est
    cost = np.sum(cost**2)
#     print(cost)
    return cost


# In[3]:


def estimate_camera_projection_matrix_linear(x, X, normalize=True):
    # Inputs:
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    normalize - if True, apply data normalization to x and X
    #
    # Output:
    #    P - the (3x4) DLT estimate of the camera projection matrix
    P = np.eye(3,4)+np.random.randn(3,4)/10
        
    # data normalization
    if normalize:
        x, T = data_normalize(x)
        X, U = data_normalize(X)
    else:
        x = homogenize(x)
        X = homogenize(X)
    
    """your code here"""
    P = get_projection_matrix(x, X) #Helper functions

    # data denormalize
    if normalize:
        P = np.linalg.inv(T) @ P @ U
    
    return P

def display_results(P, x, X, title):
    print(title+' =')
    print (P/np.linalg.norm(P)*np.sign(P[-1,-1]))

# load the data
x=np.loadtxt('hw2_points2D.txt').T
X=np.loadtxt('hw2_points3D.txt').T

assert x.shape[1] == X.shape[1]
n = x.shape[1]

# compute the linear estimate without data normalization
print ('Running DLT without data normalization')
time_start=time.time()
P_DLT = estimate_camera_projection_matrix_linear(x, X, normalize=False)
cost = sum_of_square_projection_error(P_DLT, x, X)
time_total=time.time()-time_start
# display the results
print('took %f secs'%time_total)
print('Cost=%.9f'%cost)


# compute the linear estimate with data normalization
print ('Running DLT with data normalization')
time_start=time.time()
P_DLT = estimate_camera_projection_matrix_linear(x, X, normalize=True)
cost = sum_of_square_projection_error(P_DLT, x, X)
time_total=time.time()-time_start
# display the results
print('took %f secs'%time_total)
print('Cost=%.9f'%cost)

print("\n==Correct outputs==")
print("Cost=%.9f without data normalization"%97.053718991)
print("Cost=%.9f with data normalization"%84.104680130)


# In[4]:


# Report your P_DLT (estimated camera projection matrix linear) value here!
display_results(P_DLT, x, X, 'P_DLT')


# ## Problem 4 (Programming):  Nonlinear Estimation of the Camera Projection Matrix (30 points)
#   Use $\boldsymbol{P}_\text{DLT}$ as an initial estimate to an iterative
#   estimation method, specifically the Levenberg-Marquardt algorithm,
#   to determine the Maximum Likelihood estimate of the camera
#   projection matrix that minimizes the projection error.  You must
#   parameterize the camera projection matrix as a parameterization of
#   the homogeneous vector $\boldsymbol{p} = vec{(\boldsymbol{P}^\top)}$.  It is
#   highly recommended to implement a parameterization of homogeneous
#   vector method where the homogeneous vector is of arbitrary length,
#   as this will be used in following assignments.
#   
#   Report the initial cost (i.e. cost at iteration 0) and the cost at the end
#   of each successive iteration. Show the numerical values for the final 
#   estimate of the camera projection matrix $\boldsymbol{P}_\text{LM}$, scaled
#   such that $||\boldsymbol{P}_\text{LM}||_\text{Fro} = 1$.
#   
#   The following helper functions may be useful in your LM function implementation.
#   You are welcome <i>and encouraged</i> to add any additional helper functions.
#   
#   Hint: LM has its biggest cost reduction after the 1st iteration. You'll know if you 
#   are implementing LM correctly if you experience this.

# In[5]:


import numpy as np
# Note that np.sinc is different than defined in class
def sinc(x):
    # Returns a scalar valued sinc value
    """your code here"""
    y = 1 if x == 0 else np.sin(x)/x
    
    return y

def dsinc_dx(x):
    '''
    This is a helper function to obtain the 
    derivative of sinc function
    '''
    der_val = (np.cos(x) / x) - (np.sin(x)/(x**2))
    return 0 if x == 0 else der_val

def partial_x_partial_p(P,X,x):
    # compute the dx_dp component for the Jacobian
    #
    # Input:
    #    P - 3x4 projection matrix
    #    X - Homogenous 3D scene point
    #    x - inhomogenous 2D point
    # Output:
    #    dx_dp - 2x12 matrix
    
    dx_dp = np.zeros((2,12))
    
    """your code here"""
    zero_vector = np.zeros((1, 4))
    p3_t = P[-1, :]
    w = p3_t @ X
    
    #First row
    dx_dp[0, :4] = X.T
    dx_dp[0, 4:8] = zero_vector
    dx_dp[0, 8:] = -x[0]*X.T
    
    #Second row
    dx_dp[1, :4] = zero_vector
    dx_dp[1, 4:8] = X.T
    dx_dp[1, 8:] = -x[1]*X.T
    
    dx_dp /= w
    
    return dx_dp


def parameterize_matrix(P):
    # wrapper function to interface with LM
    # takes all optimization variables and parameterizes all of them
    # in this case it is just P, but in future assignments it will
    # be more useful
    return parameterize_homog(P.reshape(-1,1))


def deparameterize_matrix(m,rows,columns):
    # Deparameterize all optimization variables
    # Input:
    #   m - matrix to be deparameterized
    #   rows - number of rows of the deparameterized matrix 
    #   columns - number of rows of the deparameterized matrix 
    #
    # Output:
    #    deparameterized matrix
    #
    # For the camera projection, deparameterize it using deparameterize_matrix(p,3,4) 
    # where p is the parameterized camera projection matrix

    return deparameterize_homog(m).reshape(rows,columns)


def parameterize_homog(v_bar):
    # Given a homogeneous vector v_bar return its minimal parameterization
    """your code here"""
    
    norm_v_bar = np.linalg.norm(v_bar)
    assert round(norm_v_bar) == 1, "Check the norm!"
    
    a = v_bar[0]
    b = v_bar[1:]
    
    v = (2 / (sinc(np.arccos(a)))) * b
    norm_v = np.linalg.norm(v)
    if norm_v > np.pi:
        norm_factor = 1 - (2*np.pi/norm_v)*np.ceil((norm_v - np.pi)/(2*np.pi))
        v = norm_factor * v
    
    v = v.reshape(-1, 1)
    return v


def deparameterize_homog(v):
    # Given a parameterized homogeneous vector return its deparameterization
    """your code here"""
    
    norm_v = np.linalg.norm(v)
    a = np.cos(norm_v/2)
    b = sinc(norm_v/2) / 2 * v
    
    v_bar = np.insert(b, 0, a).reshape(-1, 1)
    
    return v_bar

def deparameterize_homog_with_Jacobian(v):
    # Input:
    #    v - homogeneous parameterization vector (11x1 in case of p)
    # Output:
    #    v_bar - deparameterized homogeneous vector
    #    partial_vbar_partial_v - derivative of v_bar w.r.t v
    
    
    partial_vbar_partial_v = np.zeros((12,11))
    v_bar = np.zeros((12,1))
    
    """your code here"""
    v_bar = deparameterize_homog(v)
    
    a, b = v_bar[0], v_bar[1:]
    assert a >= 0, "Check value of a!" #Per lecture slides
    norm_v = np.linalg.norm(v)
    da_dv = np.zeros((1, 11)) if norm_v == 0 else -0.5*b.T
    da_dv = da_dv.reshape(1, 11) #Fail proofing
    
    I = np.eye(11)
    first_half = sinc(norm_v/2)/2 * I
    second_half = (1/(4*norm_v)) * dsinc_dx(norm_v/2) * (v @ v.T)
    db_dv = (0.5 * I) if norm_v == 0 else (first_half+second_half)
    
    partial_vbar_partial_v[0, :] = da_dv
    partial_vbar_partial_v[1:, :] = db_dv
    
    return v_bar, partial_vbar_partial_v

def data_normalize_with_cov(pts, covarx):
    # data normalization of n dimensional pts
    #
    # Input:
    #    pts - is in inhomogeneous coordinates
    #    covarx - covariance matrix associated with x. Has size 2n x 2n, where n is number of points.
    # Outputs:
    #    pts - data normalized points
    #    T - corresponding transformation matrix
    #    covarx - normalized covariance matrix
    
    """your code here"""
    

    T = np.eye(pts.shape[0]+1)
    
    mean_x = np.mean(pts[0, :])
    mean_y = np.mean(pts[1, :])
    var_x = np.var(pts[0, :])
    var_y = np.var(pts[1, :])
    
    s = np.sqrt(2/(var_x+var_y))
    T[0, 0], T[1, 1] = s, s
    T[0, 2] = -s*mean_x
    T[1, 2] = -s*mean_y
      
    pts = T @ homogenize(pts)
    
    covarx = (s**2) * covarx

    return pts, T, covarx

def compute_cost(P, x, X, covarx):
    # Inputs:
    #    P - the camera projection matrix
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    covarx - covariance matrix associated with x. Has size 2n x 2n, where n is number of points.
    # Output:
    #    cost - Total reprojection error
    
    """your code here"""

    # Code using provided helper functions
    x_hat = P @ X
    #Order="F" as the expected array is (x1, y1, x2, y2,...,x50, y50)
    eps = x.reshape(1, -1, order="F") - dehomogenize(x_hat).reshape(1, -1, order="F")
    eps = eps.T
    
    cost = eps.T @ np.linalg.inv(covarx) @ eps

    return cost


# In[6]:


#Unit Tests (Do not change)

# partial_x_partial_p unit test
def check_values_partial_x_partial_p():
    eps = 1e-8  # Floating point error threshold
    x_2d = np.load('Unit_test/x_2d.npy')
    P = np.load('Unit_test/Projection.npy')
    X = np.load('Unit_test/X.npy')
    target = np.load('Unit_test/dx_dp.npy')
    dx_dp = partial_x_partial_p(P,X,x_2d)
    valid = np.all((dx_dp < target + eps) & (dx_dp > target - eps))
    print(f'Computed partial_x_partial_p is all equal to ground truth +/- {eps}: {valid}')

# deparameterize_homog_with_Jacobian unit test
def check_values_partial_vbar_partial_v():
    eps = 1e-8  # Floating point error threshold
    p = np.load('Unit_test/p.npy')
    dp_dp_target = np.load('Unit_test/dp_dp.npy')
    p_bar_target = np.load('Unit_test/Projection.npy').reshape(12,1)
    p_bar,dp_dp = deparameterize_homog_with_Jacobian(p)
    valid_dp_dp = np.all((dp_dp < dp_dp_target + eps) & (dp_dp > dp_dp_target - eps))
    valid_p_bar = np.all((p_bar < p_bar_target + eps) & (p_bar > p_bar_target - eps))
    valid = valid_dp_dp & valid_p_bar
    print(f'Computed v_bar,partial_vbar_partial_v is all equal to ground truth +/- {eps}: {valid}')
    
check_values_partial_x_partial_p()
check_values_partial_vbar_partial_v()


# In[7]:


def get_jacobian(P, X, x_2d):
    '''
    Helper function to obtain the Jacobian at each step of LM
    '''
    _, num_points = x_2d.shape
    J = np.zeros((2*num_points, 11)) #Jacobian matrix
    p_param = parameterize_matrix(P)
    p_bar, dp_bar_dp = deparameterize_homog_with_Jacobian(p_param)
    j = 0
    for i in range(num_points):
        x_i, X_i = x_2d[:, i], X[:, i]
        dx_dp = partial_x_partial_p(P, X_i, x_i)
        J[j:j+2, :] = dx_dp @ dp_bar_dp
        j+=2
    return J
        


# In[8]:


def estimate_camera_projection_matrix_nonlinear(P, x, X, max_iters, lam):
    # Input:
    #    P - initial estimate of P
    #    x - 2D inhomogeneous image points
    #    X - 3D inhomogeneous scene points
    #    max_iters - maximum number of iterations
    #    lam - lambda parameter
    # Output:
    #    P - Final P (3x4) obtained after convergence
    
    # data normalization
    
    covarx = np.eye(2*X.shape[1])
    x, T, covarx = data_normalize_with_cov(x, covarx)
    X, U = data_normalize(X)

    """your code here"""
    #Normalize the initial DLT estimate
    P = T @ P @ np.linalg.inv(U)
    
    x_hat = dehomogenize(P @ X)
    x = dehomogenize(x)
    
    #Epsilon to calculate the inital cost
    eps = x - x_hat
    eps_reshaped = eps.reshape(1, -1, order="F").T
    cost_original = compute_cost(P, x, X, covarx)

    rows, cols = P.shape
    wrong_cost_num = 0
    tol = 1e-7
    
    # you may modify this so long as the cost is computed
    # at each iteration
    for i in range(max_iters): 
        if wrong_cost_num > 10: #Condition to break if cost is increasing
            print("Optimization is wrong, teminating...!")
            break

        p_param = parameterize_matrix(P)
        P_hat = deparameterize_matrix(p_param, rows, cols)

        J = get_jacobian(P_hat, X, x_hat)
        
        # Normal equations
        U_normal_eq = J.T @ np.linalg.inv(covarx) @ J
        S = U_normal_eq + lam * np.eye(11)
        eps_a = J.T @ np.linalg.inv(covarx) @ eps_reshaped  
        
        delta_a = np.linalg.inv(S) @ eps_a

        #Making adjustment
        p_0_param = p_param + delta_a
        P_0_hat = deparameterize_matrix(p_0_param, rows, cols)
        
        X_0_hat = P_0_hat @ X #Calculate 3D projection
        x_0_hat = dehomogenize(X_0_hat) #Dehomogenize
        eps_0 = x - x_0_hat
        eps_0_reshaped = eps_0.reshape(1, -1, order="F").T
        
        cost_outer = eps_reshaped.T @ np.linalg.inv(covarx) @ eps_reshaped
        cost = compute_cost(P_0_hat, x, X, covarx)
        
        #Loop termination criteria
        if 1-(cost/cost_outer) < tol:
            break
            
        if cost >= cost_outer:
            print("Something is wrong! Incrementing terminating criterion! - ", cost)
            wrong_cost_num += 1
            lam *= 10
            continue
        else:
            P = P_0_hat
            P /= np.linalg.norm(P)
            eps_reshaped = eps_0_reshaped
            x_hat = x_0_hat
            lam /= 10
        P /= np.linalg.norm(P)
        print ('iter %03d Cost %.9f'%(i+1, cost))
    
    # data denormalization
    P = np.linalg.inv(T) @ P @ U
    P /= np.linalg.norm(P) #Ensuring the norm is 1
    return P



# LM hyperparameters
lam = .001
max_iters = 100

# Run LM initialized by DLT estimate with data normalization
print ('Running LM with data normalization')
print ('iter %03d Cost %.9f'%(0, cost))

P_LM = estimate_camera_projection_matrix_nonlinear(P_DLT, x, X, max_iters, lam)
time_total=time.time()-time_start
print('took %f secs'%time_total)

print("\n==Correct outputs==")
print("Begins at %.9f; ends at %.9f"%(84.104680130, 82.790238005))


# In[9]:


# Report your P_LM (estimated camera projection matrix nonlinear) final value here!
display_results(P_LM, x, X, 'P_LM')


# In[ ]:




