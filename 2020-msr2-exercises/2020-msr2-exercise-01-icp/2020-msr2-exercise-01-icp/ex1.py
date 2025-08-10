#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

# icp_known_corresp: performs icp given that the input datasets
# are aligned so that Line1(:, QInd(k)) corresponds to Line2(:, PInd(k))
def icp_known_corresp(Line1, Line2, QInd, PInd):
    """
    ICP Known Correspondences
    This function performs the Iterative Closest Point (ICP) algorithm
    given that the input datasets are aligned such that Line1(:, QInd(k))
    corresponds to Line2(:, PInd(k)). It computes the optimal rotation and
    translation to align the two sets of points, and returns the new positions
    of the points after applying the found rotation and translation.

    Main Takeaways:
    - Solve the minimization problem min sum || y_n - (R * x_n + t) ||^2 * p_n
    - Optimal parameters can be analytically computed, giving the following equations:
        - R = V * U^T
        - t = y_0 - R * x_0
    """
    # Get indices corresponding to the points in Line1 and Line2
    Q = Line1[:, QInd]
    P = Line2[:, PInd]
    # print("Q ind: ", QInd)
    # print("Q: ", Q)


    MuQ = compute_mean(Q)
    MuP = compute_mean(P)
    
    W = compute_W(Q, P, MuQ, MuP)

    [R, t] = compute_R_t(W, MuQ, MuP)

    
    # Compute the new positions of the points after
    # applying found rotation and translation to them
    NewLine = R.dot(P) + t

    E = compute_error(Q, NewLine)

    return NewLine, E

# compute_W: compute matrix W to use in SVD
def compute_W(Q, P, MuQ, MuP):
    """
    Compute cross covariance matrix W with weight 1
    """
    mP = P - MuP
    mQ = Q - MuQ
    W = mQ.dot(mP.T) # considering weight as 1
    
    return W

    
# compute_R_t: compute rotation matrix and translation vector
# based on the SVD as presented in the lecture
def compute_R_t(W, MuQ, MuP):
    # add code here and remove pass
    U, S, Vt = np.linalg.svd(W)
    R = U.dot(Vt)
    if np.linalg.det(R) < 0:
        #raise ValueError("Det(R) < 0, cannot compute rotation")
        print("Warning: Det(R) < 0")
    t = MuQ - R.dot(MuP)  # Translation vector
    return R, t


# compute_mean: compute mean value for a [M x N] matrix
def compute_mean(M):
    mean = np.array([M.mean(axis=1)]).T # considerig weight as 1
    return mean


# compute_error: compute the icp error
def compute_error(Q, OptimizedPoints):
    return np.sum(np.linalg.norm(Q - OptimizedPoints)**2)/np.shape(Q)[1]


# simply show the two lines
def show_figure(Line1, Line2):
    plt.figure()
    plt.scatter(Line1[0], Line1[1], marker='o', s=2, label='Line 1')
    plt.scatter(Line2[0], Line2[1], s=1, label='Line 2')
    
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()  
    
    plt.show()
    

# initialize figure
def init_figure():
    fig = plt.gcf()
    fig.show()
    fig.canvas.draw()
    
    line1_fig = plt.scatter([], [], marker='o', s=2, label='Line 1')
    line2_fig = plt.scatter([], [], marker='o', s=1, label='Line 2')
    # plt.title(title)
    plt.xlim([-8, 8])
    plt.ylim([-8, 8])
    plt.legend()
    
    return fig, line1_fig, line2_fig


# update_figure: show the current state of the lines
def update_figure(fig, line1_fig, line2_fig, Line1, Line2, hold=False):
    line1_fig.set_offsets(Line1.T)
    line2_fig.set_offsets(Line2.T)
    if hold:
        plt.show()
    else:
        fig.canvas.flush_events()
        fig.canvas.draw()
        plt.pause(0.5)
