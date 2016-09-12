__author__ = 'kirill, larisa'
import matplotlib as mpl
mpl.use('TkAgg')

import scipy.spatial.distance as sci
import math
import matplotlib.pylab as plt

import numpy as np
import os, copy, time
import scipy.special as spec
from pylab import rcParams
import sys

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.figure import Figure
from sklearn import preprocessing, decomposition



if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk


X, weights, H, L_T, dist_matrix, step_number, true_clusters,texts, f, canvas = 0,0,0,0,0,0,0,0,0,0
n, dist_ordered, v, T, KL = 0, 0, 0, 0, 0
picked_point = 0

def onclick(event, X, weights, f, ax1, canvas, clustering):
    global picked_point
    mpl.rcParams['axes.color_cycle'] = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    f[1].clear()
    ax1 = f[1].add_subplot(111)
    #ax1.clear()
    p = [event.button, event.xdata, event.ydata]
    picked = [p[1], p[2]]
    massiv = []
    for i in xrange(len(X)):
        massiv.append(sci.euclidean(X[i, 0:2], picked))
    i = np.argmin(massiv)
    picked_point = i
    ax1.scatter(X[:, 0], X[:, 1], c=weights[i], cmap = 'YlOrRd', vmin=0, vmax=1.)
    ax1.axis('equal')
    #f.canvas.draw()
    set_title(f)
    canvas[1].show()


def k_loc(x):
    return x <= 1

def k_stat(x):
    return 1. * (x <= 0)
    return (2.- x).clip(min=0, max=1)

def distance_matrix(X):
    return sci.squareform(sci.pdist(X, 'euclidean'))

def get_neighbour_numbers(h, dist_matrix, weights):
    return np.sum(weights * k_loc(dist_matrix / h), axis = 1)

h_ratio = 1.95

def get_lambda_hash(h, d, L_T):
    global h_ratio
    #print 'a', np.max(np.min((d / h / h_ratio * 10000 ).astype(int), 10000-1))
    return np.take(L_T, np.minimum((d / h / h_ratio * (len(L_T) - 1) ).astype(int), (len(L_T) - 1)))


def get_lambda_table(n, m):
    global h_ratio
    x = np.linspace(0 + 1./m, h_ratio + 1./m, m+1)
    a = spec.betainc((n+1.) / 2, 0.5, 1. - (x/2.)**2)
    return a / (2-a)# - 1.

n_0 = 10

def initialisation(x, H):
    global n_0
    #x = x / H[0]
    dist_matrix = np.sort(x, axis=1)
    neighbor_number = copy.deepcopy(n_0)
    v = dist_matrix[:, neighbor_number - 1]
    n = np.size(x,0)
    a = np.zeros((n,n))
    for i in xrange(n):
        #a[i,:] = np.minimum(np.exp(-(x[i,:]/max(v[i], h_0)-1.) / 0.002), 1.)
        h_closest = H[-1]
        for h in H:
            if h >= v[i]:
                h_closest = h
                break
        a[i, :] = 1 * (x[i,:] <= h_closest) 
    a = np.maximum(a, a.T)
    #a = a.astype(bool)
    #return np.ones((n,n))
    return a
cid = 0.1

cluster_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'gold', 'firebrick', 'olive', 'springgreen', 'palevioletred', 'hotpink','lightgreen']
    
def draw_step(f, canvas, weights, X, true_clusters=[], clustering=False, depth = 0):
    global cid, picked_point, X_PCA
    f[0].clear()
    f[1].clear()
    f[2].clear()
    f[3].clear()
    global cluster_colors
    n = np.size(X_PCA, 0)
    
    # f.add_subplot(221)
    ax0,ax1,ax2,ax3 = f[0].add_subplot(111),f[1].add_subplot(111),f[2].add_subplot(111),f[3].add_subplot(111),
    #((ax1, ax2), (ax3, ax4)) = f.add_subplot(221)
    ax0.imshow(weights, cmap=plt.cm.gray, interpolation='nearest')
    
    ax1.scatter(X_PCA[:, 0], X_PCA[:, 1], c=weights[picked_point], cmap = 'YlOrRd', vmin=0, vmax=1.)
    #ax1.plot(X[:, 0], X[:, 1], 'go')
    
    
    
    adjmatrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if weights[i, j] >= 0.5:
                adjmatrix[i, j] = 1
    colors = np.zeros((n,))
    for cluster_i in range(len(true_clusters)):
        #print 'h', len(true_clusters)
        cluster = true_clusters[cluster_i]
        x = X_PCA[cluster, :]
        colors[cluster] = cluster_i
        #print 'o' + cluster_colors[cluster_i]
        ax3.plot(x[:, 0], x[:, 1],  marker='o', linestyle='None', color=cluster_colors[cluster_i])      
    
    if clustering:
        U = copy.deepcopy(X_PCA)
        points = range(len(X_PCA))
        not_used_colors = range(len(cluster_colors))
        while(np.size(U, 0) != 0):
            neigbohours = np.sum(adjmatrix, axis=1)
            candidates = np.argsort(neigbohours)
            #candidates = np.argsort(X[points, 0])
            for cluster_generater in reversed(candidates):#xrange(len(U)):
                neigbohours_i = neigbohours[adjmatrix[cluster_generater, :] == 1]
                if np.sum(neigbohours_i > (neigbohours[cluster_generater] - 5)) > 0.9 * np.size(neigbohours_i,0):
                    break
            all_cliques = ()
            for i in range(len(U)):
                if adjmatrix[cluster_generater, i] == 1:
                    all_cliques += tuple([i])
            all_cliques = [all_cliques]
            if len(not_used_colors) > 0:
                colors_n = np.zeros((len(not_used_colors),))
                for k in range(len(colors_n)):
                    colors_n[k] = len(colors[colors[list(all_cliques[0]),] == not_used_colors[k]])
                true_color = np.argmax(colors_n)
                #print 'o' + cluster_colors[not_used_colors[true_color]]
                ax2.plot(U[all_cliques[0], 0], U[all_cliques[0], 1],  marker='o', linestyle='None', color=cluster_colors[not_used_colors[true_color]])
                #ax.scatter(U[all_cliques[0], 0], U[all_cliques[0], 1], U[all_cliques[0], 2],zdir='z', c=cluster_colors[not_used_colors[true_color]])
                del not_used_colors[true_color]
                colors = np.delete(colors, all_cliques[0], 0)
            else:
                ax2.plot(U[all_cliques[0], 0], U[all_cliques[0], 1], 'o')
                #ax.scatter(U[all_cliques[0], 2], U[all_cliques[0], 0], U[all_cliques[0], 1],zdir='z')
            U = np.delete(U, all_cliques[0], 0)  
            adjmatrix = np.delete(adjmatrix, all_cliques[0], 0)
            adjmatrix = np.delete(adjmatrix, all_cliques[0], 1)
            #print all_cliques[0]
            #print points
            points = [points[i] for i in range(len(points)) if i not in all_cliques[0]]
    #plt.show()
    if depth == 0:
        if cid != 0.1:
            f[1].canvas.mpl_disconnect(cid)
        cid = f[1].canvas.callbacks.connect('button_press_event', lambda event: onclick(event, X_PCA, weights, f, ax1, canvas, clustering))
        #print f[1].canvas.mpl_connect('button_press_event', lambda event: onclick(event, X, weights, f, ax1, canvas, clustering))
    ax1.axis('equal')
    ax2.axis('equal')
    ax3.axis('equal')
    #f.plot()
    #plt.show()
    set_title(f)
    canvas[0].show()
    canvas[1].show()
    canvas[2].show()
    canvas[3].show()
    return 0
    
    
def KL_init():
    m = 2000
    e1 = np.linspace(0, 1, m + 1)
    e = np.repeat([e1], m + 1, axis=0)
    q = e.T
    #print e[0, :]
    #print q[0, :]
    KL = (-1) ** (e > q) * (e-q) * np.log((e * (1. - q) / q / (1. - e)))
    #KL = ((e-q) * np.log((e / q)) + ((q - e) ) * np.log(  (1 - e) / (1 - q)  )  )
    #KL = (-1) ** (e > q) * (e * np.log(e / q) + (1 - e) * np.log(  (1 - e) / (1 - q)  )  )
    #KL = (-1) ** (e > q) * ((-q) * np.log((e / q)) + ((-1 + q) ) * np.log(  (1 - e) / (1 - q)  )  )
    #KL[np.isnan(KL)] = 0 
    KL = np.nan_to_num(KL)
    #print KL[0,1]
    print 'sss', KL[-1, 0]
    return KL
    
def init(X, n_neigh):
    global n_0#, weights_init
    if n_neigh == -1:
        n_0 = 2 * np.size(X, 1) + 2
    print 'yyyy', n_0, n_neigh
    rcParams['figure.figsize'] = 10, 10
    rcParams['figure.figsize'] = 8, 6
    n = len(X)
    L_T = get_lambda_table(np.size(X,1), 10000)
    
    dist_matrix = distance_matrix(X)
    H, dist_ordered = get_h_intervals(dist_matrix)
    print 'H', len(H)
    v = dist_ordered[:, n_0-1].clip(min=H[0])
    #weights = np.zeros((n, n))
    weights = initialisation(dist_matrix, H)  
    #weights_init = copy.deepcopy(weights)  
    #flag = np.zeros((n,n), dtype=np.int8)
    T = np.zeros((n, n))
    KL = KL_init()
    return n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL


def cluster(f, canvas,  l, show_step = False, show_finish = False, T_stat_show = False, log_show = True, n_neigh = -1, method = 1, step=None, clustering = False):
    #print np.size(X,0),
    global X, true_clusters, n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL
    rcParams['figure.figsize'] = 10, 10
    
    for k in range(len(H)):
        if show_step:
            draw_step(f,canvas,weights, X, true_clusters, clustering)
        neighbour_numbers = np.sum(weights * (dist_matrix <= H[k-1]), axis = 1) - 1
        D2 = dist_matrix <= H[k-1]
        np.fill_diagonal(D2, False)
        P =  D2 * weights
        #max_dist = np.max(dist_matrix, axis=1)
        
        max_dist = np.max(dist_matrix, axis=1)
        
        t_1 = (neighbour_numbers[np.newaxis, :] - P).T
        
        t_12 = np.inner(P, P)
        t_12x = np.inner(P, D2)
        #print 't_1=', t_1[x, y]
        gg1 = (t_1 == t_12x) * (t_12 < 0.5 * t_12x)
        #gg2 = (t_1.T == t_12x.T) * (t_12 < 0.5 * t_12x.T)
        
        t_1 = t_1 - t_12x + t_12
        q = get_lambda_hash(H[k-1], dist_matrix, L_T)
        #E = (max_dist[i] < H[k-1]) * ( max_dist[i+1:] < H[k-1])
        #q[E] = 1. / get_lambda_hash(np.maximum(max_dist[i], max_dist[i+1:][E]), dist_matrix[i, i+1:][E], L_T)
        
        E = max_dist < H[k-1]
        F = np.repeat([max_dist], n, axis = 0)
        R = np.maximum(F.T, F)
        q[E, :][:, E] = get_lambda_hash(R[E, :][:, E], dist_matrix[E, :][:, E], L_T)
        
        
        t = t_1 + t_1.T - t_12
        e = t_12 / t
        #print 't_12x=', t_12x[x, y], 't_21x=', t_12x[y, x], 't_1=', t_1[x, y]
        #print 't_2=', t_1[y, x], 't_12=', t_12[x, y], 'e=', e[x, y], 'q=', q[x, y]
        
        aa = e >= 0.95
        e[t == 0] = 0
        e = e.clip(min=0.05, max=0.9)
        q = q.clip(min=0.05, max=0.9)
        bb = e <= 0.05
        e *= 2000
        q *= 2000
        
        e = e.astype(int)
        q = q.astype(int)
        
        T = t * KL[q, e]
        T[np.logical_or(bb, t_12 == 0)] = np.nan
        T[aa] = l
        sum_v = v > H[k-1]
        T[sum_v, :] = float("inf")
        T[:, sum_v] = float("inf")
        T[np.logical_or(gg1, gg1.T)] = np.nan
        #T[gg2] = np.nan
        
        #print 'T', T[x, y], T[y,x]
        ###print 'T[' + str(x) + ',' + str(y) + ']', T[x, y]
        #start_time = time.time()
        I = (dist_matrix <= H[k]) * (dist_matrix > 0) * (T != float("inf")) * (np.isnan(T) == False)
        weights[I] = 1 * (T[I] <= l)
        start_time = time.time()
        weights[np.isnan(T)] = 0
        np.fill_diagonal(weights, 1)
    if show_finish:
        draw_step(f,canvas,weights, X, true_clusters, clustering)
    return weights

def cluster_step(f, canvas, l, clustering = False):
    #print 'l', l
    global X, true_clusters, n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL, step_number
    global weights_computed
    #print 'v', (weights_computed[0] - weights)[(weights_computed[0] - weights) != 0]
    n = len(X)
    #if step_number == 0:
    #	n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL = init(X, n_neigh=-1)
    #    rcParams['figure.figsize'] = 10, 10
    #    T = np.zeros((n, n))
    if step_number >= len(H):
        print 'ererere'
        return 0
    
    for k in [step_number]:
        weights = copy.deepcopy(weights_computed[step_number - 1])
        print 'k', k
        neighbour_numbers = np.sum(weights * (dist_matrix <= H[k-1]), axis = 1) - 1
        D2 = dist_matrix <= H[k-1]
        np.fill_diagonal(D2, False)
        P =  D2 * weights
        #max_dist = np.max(dist_matrix, axis=1)
        
        max_dist = np.max(dist_matrix, axis=1)
        x = 246
        y = 264
        
        
        t_1 = (neighbour_numbers[np.newaxis, :] - P).T
        
        t_12 = np.inner(P, P)
        t_12x = np.inner(P, D2)
        #print 't_1=', t_1[x, y]
        gg1 = (t_1 == t_12x) * (t_12 < 0.5 * t_12x)
        #gg2 = (t_1.T == t_12x.T) * (t_12 < 0.5 * t_12x.T)
        
        t_1 = t_1 - t_12x + t_12
        q = get_lambda_hash(H[k-1], dist_matrix, L_T)
        #E = (max_dist[i] < H[k-1]) * ( max_dist[i+1:] < H[k-1])
        #q[E] = 1. / get_lambda_hash(np.maximum(max_dist[i], max_dist[i+1:][E]), dist_matrix[i, i+1:][E], L_T)
        
        E = max_dist < H[k-1]
        F = np.repeat([max_dist], n, axis = 0)
        R = np.maximum(F.T, F)
        q[E, :][:, E] = get_lambda_hash(R[E, :][:, E], dist_matrix[E, :][:, E], L_T)
        
        
        t = t_1 + t_1.T - t_12
        e = t_12 / t
        #print 't_12x=', t_12x[x, y], 't_21x=', t_12x[y, x], 't_1=', t_1[x, y]
        #print 't_2=', t_1[y, x], 't_12=', t_12[x, y], 'e=', e[x, y], 'q=', q[x, y]
        
        aa = e >= 0.95
        e[t == 0] = 0
        e = e.clip(min=0.05, max=0.9)
        q = q.clip(min=0.05, max=0.9)
        bb = e <= 0.05
        e *= 2000
        q *= 2000
        
        e = e.astype(int)
        q = q.astype(int)
        
        T = t * KL[q, e]
        T[np.logical_or(bb, t_12 == 0)] = np.nan
        T[aa] = l
        sum_v = v > H[k-1]
        T[sum_v, :] = float("inf")
        T[:, sum_v] = float("inf")
        T[np.logical_or(gg1, gg1.T)] = np.nan
        #T[gg2] = np.nan
        
        #print 'T', T[x, y], T[y,x]
        ###print 'T[' + str(x) + ',' + str(y) + ']', T[x, y]
        #start_time = time.time()
        I = (dist_matrix <= H[k]) * (dist_matrix > 0) * (T != float("inf")) * (np.isnan(T) == False)
        weights[I] = 1. * (T[I] <= l)
        start_time = time.time()
        weights[np.isnan(T)] = 0
        np.fill_diagonal(weights, 1)
        #print 'w', weights[0, :]
    #draw_step(f,canvas, weights, X, true_clusters, clustering)
    step_number += 1
    return 0

def get_h_intervals(dist_matrix, log_show=False):
    global n_0, h_ratio
    #print 'n_0=', n_0
    print '1', dist_matrix[0, :]
    dist_matrix = np.sort(dist_matrix)
    h_intervals = [np.percentile(dist_matrix[:, n_0 - 1], 30)]
    neighbor_number = copy.deepcopy(n_0)
    neighbor_number *= 2 ** 0.5
    #plt.plot(range(len(h_intervals)), h_intervals)
    #plt.show()
    #return h_intervals, dist_matrix
    ### New idea  
    
    ### Another Idea
    neighbor_number_seq = [n_0]
    while(1):
        a = int(neighbor_number_seq[-1] * 2 ** 0.3)
        if a < np.size(dist_matrix, 0)-1:
            neighbor_number_seq.append(a)
        else:
            a = np.size(dist_matrix, 0) - 1
            neighbor_number_seq.append(a)
            break
    #print 'NNN', neighbor_number_seq
    
    h_intervals = np.reshape(dist_matrix[:, neighbor_number_seq], (-1))
    
    indexes = copy.deepcopy((dist_matrix))
    for i in xrange(np.size(indexes,0)):
        indexes[i, :] = i
    indexes_h = np.reshape(indexes[:, neighbor_number_seq], (-1))
    permutation_sort = np.argsort(h_intervals)
    
    #h_intervals = h_intervals[permutation_sort]
    indexes_h = indexes_h[permutation_sort]
    indexes_h = indexes_h.astype(int)
    
    print 'RRRRRRRRR'
    
    h_final = []
    stack = []
    n_break_points = 0
    break_level = np.size(dist_matrix, 0) / 10
    for i in range(len(h_intervals)):
        nn = stack.count(indexes_h[i])
        if nn < 2 and n_break_points < break_level:
            stack.append(indexes_h[i])
            if nn == 1:
                n_break_points += 1
        else:
            h_final.append(h_intervals[permutation_sort[i]])
            stack = []
            n_break_points = 0
        if i == len(h_intervals)-1:
            h_final.append(h_intervals[permutation_sort[i]])
            break
    print 'RRRRRRRRR'
    print len(h_final), len(h_intervals), h_final
    #return h_final, dist_matrix
    a = [h_final[0]]
    for i in xrange(len(h_final)-1):
        if h_final[i+1] - h_final[i] != 0:
            a.append(h_final[i+1])
    #a.append(a[-1])
    #a = a + a
    if a[0] == 0:
        del a[0]
    for i in range(6):
        a.append(a[-1]* 1.5)
    print 'a', a
    return a, dist_matrix


def generate_two_normal(distance = 4, dim = 2, n_1 = 100, n_2 = 100, m_1=None, m_2=None, sigma_1 = None, sigma_2 = None):
	if sigma_1 is None:
		sigma_1 = np.identity(dim)
	if sigma_2 is None:
		sigma_2 = np.identity(dim)
	if m_1 is None:
		m_1 = np.zeros((dim,))
	if m_2 is None:
		m_2 = np.zeros((dim,))
		m_2[0] += distance		
	X = np.random.multivariate_normal(m_1, sigma_1, n_1)
	Y = np.random.multivariate_normal(m_2, sigma_2, n_2)
	X = np.concatenate((X, Y), axis = 0)
	Y_mean = np.zeros((dim,))
	Y_mean[0] = distance
	density_1 = np.zeros((n_1 + n_2, 1))
	density_2 = np.zeros((n_1 + n_2, 1))
	for i in xrange(n_1 + n_2):
		density_1[i] = normal_density(m_1, sigma_1, X[i,:])
		density_2[i] = normal_density(m_2, sigma_2, X[i,:])
	true_cluster_1 = [i for i in range(n_1 + n_2) if density_1[i] > density_2[i]]
	true_cluster_2 = [i for i in range(n_1 + n_2) if density_1[i] <= density_2[i]]
	#true_cluster_1 = [i for i in range(2 * n) if X[i, 0] < distance / 2.]
	#true_cluster_2 = [i for i in range(2 * n) if X[i, 0] >= distance / 2.]
	return X, [true_cluster_1, true_cluster_2]

def generate_three_normal(distance = 4, dim = 2, n_1 = 100, n_2 = 100, n_3=100, m_1=None, m_2=None, m_3=None, sigma_1 = None, sigma_2 = None, sigma_3 = None):
	distance /= math.sqrt(2)
	if sigma_1 is None:
		sigma_1 = np.identity(dim)
	if sigma_2 is None:
		sigma_2 = np.identity(dim)
	if sigma_3 is None:
		sigma_3 = np.identity(dim)
	if m_1 is None:
		m_1 = np.zeros((dim,))
		m_1[1] += distance
	if m_2 is None:
		m_2 = np.zeros((dim,))
		m_2[0] += -distance * math.sqrt(3) / 2.
		m_2[1] += -distance / 2.
	if m_3 is None:
		m_3 = np.zeros((dim,))
		m_3[0] += distance * math.sqrt(3) / 2.
		m_3[1] += -distance / 2.
	X = np.random.multivariate_normal(m_1, sigma_1, n_1)
	Y = np.random.multivariate_normal(m_2, sigma_2, n_2)
	Z = np.random.multivariate_normal(m_3, sigma_3, n_3)
	X = np.concatenate((X, Y, Z), axis = 0)
	density_1 = np.zeros((n_1 + n_2+ n_3, 1))
	density_2 = np.zeros((n_1 + n_2+ n_3, 1))
	density_3 = np.zeros((n_1 + n_2+ n_3, 1))
	for i in xrange(n_1 + n_2+ n_3):
		density_1[i] = normal_density(m_1, sigma_1, X[i,:])
		density_2[i] = normal_density(m_2, sigma_2, X[i,:])
		density_3[i] = normal_density(m_3, sigma_3, X[i,:])
	true_cluster_1 = [i for i in range(n_1 + n_2+ n_3) if density_1[i] >= density_2[i] and density_1[i] >= density_3[i]]
	true_cluster_2 = [i for i in range(n_1 + n_2+ n_3) if density_2[i] > density_1[i] and density_2[i] >= density_3[i]]
	true_cluster_3 = [i for i in range(n_1 + n_2+ n_3) if density_3[i] > density_1[i] and density_3[i] > density_2[i]]
	return X, [true_cluster_1, true_cluster_2, true_cluster_3]

def get_error(weights, true_weights, separate_errors = False):
	error_1 = np.sum(np.abs(weights) * (true_weights == 0)) / np.sum(np.abs(1 - true_weights) * (true_weights == 0))
	error_2 = np.sum(np.abs(weights - 1) * (true_weights == 1)) / np.sum(np.abs(np.identity(np.size(weights,0)) - 1) * (true_weights == 1))
		
	if separate_errors:
		return error_1, error_2
	else:
		return (np.sum(np.abs(weights) * (true_weights == 0)) + np.sum(np.abs(weights - 1) * (true_weights == 1))) / np.size(weights,0) / (np.size(weights,0)-1)

def get_error_cluster(weights, true_clusters, separate_errors = False):
	n = sum([len(cluster_i) for cluster_i in true_clusters])
	true_weights = np.zeros((n,n))
	for cluster_i in true_clusters:
		for i in cluster_i:
			for j in cluster_i:
				true_weights[i,j] = 1.
	error_1 = np.sum(np.abs(weights) * (true_weights == 0)) / np.sum(np.abs(1 - true_weights) * (true_weights == 0))
	error_2 = np.sum(np.abs(weights - 1) * (true_weights == 1)) / np.sum(np.abs(np.identity(np.size(weights,0)) - 1) * (true_weights == 1))
	if separate_errors:
		return error_1, error_2
	else:
		return (np.sum(np.abs(weights) * (true_weights == 0)) + np.sum(np.abs(weights - 1) * (true_weights == 1))) / np.size(weights,0) / (np.size(weights,0)-1)


def show_lambda_error(dim, n_array, n_repetition, lamda = np.linspace(1.2,1.7, 10)):
	for n in n_array:
		print ""
		print "n=", n,
		error = np.zeros((len(lamda),))
		for repeat in range(n_repetition):
			print repeat,
			true_weights = np.ones((n, n))
			X = []
			while(1):
				x = np.random.uniform(-1,1,(1, dim))
				if True:#np.linalg.norm(x) <= 1:
					if X == []:
						X = x 
					else:
						X = np.concatenate((X, x), axis = 0)
						if np.size(X, 0) == n:
							break
			true_clusters = [range(n)]
			for i in xrange(len(lamda)):
				error[i] += get_error(cluster(X, lamda[i], true_clusters), true_weights, separate_errors =True)[1]
		error /= n_repetition
		plt.plot(lamda, error, linewidth = 4.0, label="n=" + str(n))
	plt.rcParams.update({'font.size': 25})
	plt.xlabel('$\lambda$', fontsize = 55)
	plt.ylabel('$error$', fontsize = 55)
	plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
	plt.legend(loc=1)
	plt.show()
	return 0

def separation_test(n_clusters = 2, dim=2, n=30, n_repetition=2, lamda = 1.5, cluster_distance = np.linspace(0, 8, 15), sigma_1 = None, sigma_2 = None, sigma_3 = None):
	if sigma_1 is None:
		sigma_1 = np.identity(dim)
	if sigma_2 is None:
		sigma_2 = np.identity(dim)
	if sigma_3 is None:
		sigma_3 = np.identity(dim)
	cl_n = len(cluster_distance)
	error_average_1 = np.zeros((cl_n, 1))
	error_average_2 = np.zeros((cl_n, 1))
	for sample_i in xrange(n_repetition):
		X_1 = np.random.multivariate_normal(np.zeros((dim,)), sigma_1, n)
		X_2 = np.random.multivariate_normal(np.zeros((dim,)), sigma_2, n)
		X_3 = np.random.multivariate_normal(np.zeros((dim,)), sigma_2, n)
		for distance_i in xrange(cl_n):
			true_weights = np.zeros((n_clusters * n, n_clusters * n))
			if n_clusters == 2:
				Y = copy.deepcopy(X_2)
				Y[:, 0] += cluster_distance[distance_i]
				X = np.concatenate((X_1, Y), axis = 0)
				true_cluster_1 = [i for i in range(2 * n) if X[i, 0] < cluster_distance[distance_i] / 2.]
				true_cluster_2 = [i for i in range(2 * n) if X[i, 0] >= cluster_distance[distance_i] / 2.]
				true_clusters = [true_cluster_1, true_cluster_2]
				for i in true_clusters[0]:
					for j in true_clusters[0]:
						true_weights[i,j] = 1
				for i in true_clusters[1]:
					for j in true_clusters[1]:
						true_weights[i,j] = 1
			if n_clusters == 3:
				X = copy.deepcopy(X_1)
				Y = copy.deepcopy(X_2)
				Z = copy.deepcopy(X_3)
				X[:, 1] += cluster_distance[distance_i] / math.sqrt(2)
				Y[:, 0] += -cluster_distance[distance_i] * math.sqrt(2) / 2. / math.sqrt(2)
				Y[:, 1] += -cluster_distance[distance_i] / 2. / math.sqrt(2)
				Z[:, 0] += cluster_distance[distance_i] * math.sqrt(2) / 2. / math.sqrt(2)
				Z[:, 1] += -cluster_distance[distance_i] / 2. / math.sqrt(2)
				X_mean = np.zeros((dim,))
				Y_mean = np.zeros((dim,))
				Z_mean = np.zeros((dim,))
				X_mean[1] += cluster_distance[distance_i]
				Y_mean[0] += -cluster_distance[distance_i] * math.sqrt(2) / 2.
				Y_mean[1] += -cluster_distance[distance_i] / 2.
				Z_mean[0] += cluster_distance[distance_i] * math.sqrt(2) / 2.
				Z_mean[1] += -cluster_distance[distance_i] / 2.
				X = np.concatenate((X, Y, Z), axis = 0)
				density_1 = np.zeros((3 *n, 1))
				density_2 = np.zeros((3 *n, 1))
				density_3 = np.zeros((3 *n, 1))
				for i in xrange(3 * n):
					density_1[i] = normal_density(X_mean, sigma_1, X[i,:])
					density_2[i] = normal_density(Y_mean, sigma_2, X[i,:])
					density_3[i] = normal_density(Z_mean, sigma_3, X[i,:])
				true_cluster_1 = [i for i in range(3 * n) if density_1[i] >= density_2[i] and density_1[i] >= density_3[i]]
				true_cluster_2 = [i for i in range(3 * n) if density_2[i] > density_1[i] and density_2[i] >= density_3[i]]
				true_cluster_3 = [i for i in range(3 * n) if density_3[i] > density_1[i] and density_3[i] > density_2[i]]
				true_clusters = [true_cluster_1, true_cluster_2, true_cluster_3]
				for i in true_clusters[0]:
					for j in true_clusters[0]:
						true_weights[i,j] = 1
				for i in true_clusters[1]:
					for j in true_clusters[1]:
						true_weights[i,j] = 1
				for i in true_clusters[2]:
					for j in true_clusters[2]:
						true_weights[i,j] = 1
			print 'a'
			print distance_i,
			weights = cluster(X, lamda, true_clusters, show_finish=False)
			print np.shape(true_weights), np.shape(X)
			error_1, error_2 = get_error(weights, true_weights, separate_errors=True)
			print error_1, error_2
			error_average_1[distance_i] += error_1
			error_average_2[distance_i] += error_2
	error_average_1 /= 1. * n_repetition
	error_average_2 /= 1. * n_repetition
	plt.plot(cluster_distance, error_average_1, label = 'union')
	plt.plot(cluster_distance, error_average_2, label = 'propagation')
	#plt.plot(cluster_distance, error_average_1 + error_average_2, label = str(dim_i))
	plt.xlabel('cluster_distance',fontsize=24)
	plt.ylabel('error',fontsize=24)
	plt.legend()
	plt.title('$\lambda=$' + str(lamda) + ', N =' + str(n) + '+' + str(n),fontsize=24)
	plt.show()
	return 0

def normal_density(m, sigma, x):
	if np.size(m,0) != 1:
		m = m.T
	if np.size(x,0) != 1:
		x = x.T
	k = len(m)
	return 1 / ((2 * math.pi) ** k / np.linalg.det(sigma) ) ** 0.5 * math.exp(-0.5 * np.dot(np.dot((x-m), np.linalg.inv(sigma)), (x-m).T))
		
def generate_orange():
	X = []
	r = np.linspace(4,5.5,4)
	dens = 10
	for r_i in r:
		N_i = r_i * dens
		points_i = np.linspace(0, 2 * math.pi * r_i, N_i) + np.random.uniform(0, 1, 1)
		for point in points_i:
			x = r_i * math.cos(point / r_i)
			y = r_i * math.sin(point / r_i)
			if X == []:
				print x, y
				X = np.zeros((1,2))
				X[0, 0], X[0, 1] = x, y
			else:
				X = np.concatenate((X, [[x, y]]), axis=0)
	
	N1 = np.size(X, 0)
	true_clusters = [range(N1)]
				
	r = np.linspace(0.01,2,7)
	dens = 20
	for r_i in r:
		N_i = r_i * dens
		points_i = np.linspace(0, 2 * math.pi * r_i, N_i) + np.random.uniform(0, 1, 1)
		for point in points_i:
			x = r_i * math.cos(point / r_i)
			y = r_i * math.sin(point / r_i)
			if X == []:
				print x, y
				X = np.zeros((1,2))
				X[0, 0], X[0, 1] = x, y
			else:
				X = np.concatenate((X, [[x, y]]), axis=0)
	true_clusters += [range(N1, np.size(X, 0), 1)]
	return X, true_clusters

def refresh(f, canvas, show_step, w, clustering,texts,show_movie):
    print '55555'
    global step_number, X, true_clusters, weights, weights_computed, K_max,lamda_picked
    
    for i in xrange(len(weights_computed)):
        if weights_computed[i] == None:
            step_number = i
            break
        step_number = i
    
    remember_step = w[6].get()
    
    if w[1].get() != lamda_picked:
        #print 'qwerty'
    	get_number_of_steps()
    if weights_computed[0] == None:
        weights_computed[0] = copy.deepcopy(weights)
        step_number = 1
    #step_number += 1
    w[6].set(remember_step)
    	
    if show_movie.get() == 1:
        #print '121212'
    	for step in range(w[6].get(), K_max, 1):
    		if weights_computed[step] is None:
    			cluster_step(f, canvas, w[1].get(), clustering = clustering.get())
    			weights_computed[step] = copy.deepcopy(weights)
    			error_1, error_2 = get_error_cluster(weights_computed[step], true_clusters, separate_errors = True)
    			#print error_1, error_2
    			texts[1].delete(1.0, Tk.END)
    			texts[2].delete(1.0, Tk.END)
    			texts[1].insert(Tk.INSERT, "{:.2f}".format(error_1) )
    			texts[2].insert(Tk.INSERT, "{:.9f}".format(error_2))
    		draw_step(f,canvas, weights_computed[step], X, true_clusters, clustering=clustering.get())
    		w[6].set(w[6].get() + 1)
    elif show_step.get() == 0:
        #print '00000'
    	for step in range(1, K_max):
            print 'step', step
            weights = copy.deepcopy(weights_computed[0])
            w[6].set(w[6].get() + 1)
            if weights_computed[step] is None:
                step_number = step
            	cluster_step(f, canvas, w[1].get(), clustering = False)
            	weights_computed[step] = copy.deepcopy(weights)
    		
    		#print w[6].get()
    	draw_step(f,canvas, weights_computed[w[6].get()], X, true_clusters, clustering=clustering.get())
    	#weights = cluster(f, canvas, w[1].get() ,show_step = False, show_finish = True, T_stat_show = False, clustering = clustering.get())
    else:
        #print '2222'
        #print w[6].get(),  step_number
    	if w[6].get() != step_number-1:
    		print '6666'
    		if weights_computed[w[6].get()] is None:
    			for i in range(step_number, w[6].get()+1,1):
    				#print "i", i
    				cluster_step(f, canvas, w[1].get(), clustering = (clustering.get() and i== w[6].get()))
    				weights_computed[i] = copy.deepcopy(weights)
    			draw_step(f,canvas, weights_computed[w[6].get()], X, true_clusters, clustering=clustering.get())
    			w[6].set(w[6].get() + 1)
    		else:
    			draw_step(f,canvas, weights_computed[w[6].get()], X, true_clusters, clustering=clustering.get())
    			w[6].set(w[6].get() + 1)
    	else:
            if step_number == 0:
                #print '7777'
            	cluster_step(f, canvas, w[1].get(), clustering = clustering.get())
            	weights_computed[0] = copy.deepcopy(weights)
            	draw_step(f,canvas, weights, X, true_clusters, clustering=clustering.get())
            	step_number = 1
            	w[6].set(1)
            	cluster_step(f, canvas, w[1].get(), clustering = clustering.get())
            	weights_computed[1] = copy.deepcopy(weights)
            else:
                #print '8888'
            	draw_step(f,canvas, weights, X, true_clusters, clustering = clustering.get())
            	
            	w[6].set(step_number)
            	cluster_step(f, canvas, w[1].get(), clustering = clustering.get())
                weights_computed[step_number-1] = copy.deepcopy(weights)
    error_1, error_2 = get_error_cluster(weights_computed[w[6].get()-1], true_clusters, separate_errors = True)
    #print error_1, error_2
    texts[1].delete(1.0, Tk.END)
    texts[2].delete(1.0, Tk.END)
    texts[1].insert(Tk.INSERT, "{:.2f}".format(error_1) )
    texts[2].insert(Tk.INSERT, "{:.9f}".format(error_2))
    return 0

lamda_N   = [20,   50, 100, 200,300, 400, 500, 600, 800, 1000, 1500]
lamda_rec = [1.0, 1.2, 1.35,1.4,1.45,1.47,1.47,1.5, 1.55, 1.55, 1.55]


def get_2_normal(f, canvas,w):
    global X, true_clusters, S
    sigma1 = [[float(S[1][0].get("1.0",Tk.END)),float(S[1][1].get("1.0",Tk.END))], [float(S[1][2].get("1.0",Tk.END)),float(S[1][3].get("1.0",Tk.END))]]
    sigma2 = [[float(S[2][0].get("1.0",Tk.END)),float(S[2][1].get("1.0",Tk.END))], [float(S[2][2].get("1.0",Tk.END)),float(S[2][3].get("1.0",Tk.END))]]
    
    X, true_clusters = generate_two_normal(distance=w[2].get(), dim=2, n_1=w[3].get(), n_2=w[4].get(),sigma_1=sigma1, sigma_2=sigma2)
    
    
    global X_PCA
    if np.size(X, 1) == 2:
        X_PCA = copy.deepcopy(X)
    else:
        n_components = 2
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(X)
        X_PCA = pca.transform(X)
    f[3].clear()
    ax3 = f[3].add_subplot(111)
    for cluster in true_clusters:
    	x = X_PCA[cluster, :]
    	ax3.plot(x[:, 0], x[:, 1], 'o')
    ax3.axis('equal')
    canvas[3].show()
    #set_lamda()
    get_number_of_steps()
    w[1].set(5)
    return 0

def get_3_normal(f, canvas,w):
    global X, true_clusters, S
    sigma1 = [[float(S[1][0].get("1.0",Tk.END)),float(S[1][1].get("1.0",Tk.END))], [float(S[1][2].get("1.0",Tk.END)),float(S[1][3].get("1.0",Tk.END))]]
    sigma2 = [[float(S[2][0].get("1.0",Tk.END)),float(S[2][1].get("1.0",Tk.END))], [float(S[2][2].get("1.0",Tk.END)),float(S[2][3].get("1.0",Tk.END))]]
    sigma3 = [[float(S[3][0].get("1.0",Tk.END)),float(S[3][1].get("1.0",Tk.END))], [float(S[3][2].get("1.0",Tk.END)),float(S[3][3].get("1.0",Tk.END))]]
    X, true_clusters = generate_three_normal(distance=w[2].get(), dim=2, n_1=w[3].get(), n_2=w[4].get(), n_3=w[5].get(), sigma_1=sigma1, sigma_2=sigma2,sigma_3=sigma3)
    global X_PCA
    if np.size(X, 1) == 2:
        X_PCA = copy.deepcopy(X)
    else:
        n_components = 2
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(X)
        X_PCA = pca.transform(X)
    f[3].clear()
    ax3 = f[3].add_subplot(111)
    for cluster in true_clusters:
    	x = X_PCA[cluster, :]
    	ax3.plot(x[:, 0], x[:, 1], 'o')
    ax3.axis('equal')
    canvas[3].show()
    #set_lamda()
    get_number_of_steps()
    w[1].set(5)
    return 0
	
def set_lamda():
	global X, w, lamda_N, lamda_rec
	n = len(X)
	if n < lamda_N[0]:
		w[1].set(lamda_rec[0])
		return 0
	if n > lamda_N[-1]:
		w[1].set(lamda_rec[-1])
		return 0
	for i in range(len(lamda_N)-1):
		if lamda_N[i] <= n and n <= lamda_N[i+1]:
			l_1 = lamda_rec[i]
			l_2 = lamda_rec[i+1]
			x_1 = lamda_N[i]
			x_2 = lamda_N[i+1]
			x_3 = n
			w[1].set(l_1 + (l_2 - l_1) / (x_2 - x_1) * (x_3 - x_1))
	
def set_title(f):
	f[0].suptitle("AWC weights")
	f[1].suptitle("AWC weights for one point")
	f[2].suptitle("AWC clustering")
	f[3].suptitle("true/wanted clustering")
	f[4].suptitle("true/wanted weights")
	return 0

K_max, weights_computed, lamda_picked = 0,0,0

def get_number_of_steps():
    global X, w, K_max, root,V, picked_point, step_number,weights_computed, lamda_picked, true_clusters, f, canvas
    lamda_picked = w[1].get()
    step_number = 0
    picked_point = 0
    global n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL
    n, L_T, dist_matrix, H, dist_ordered, v, weights, T, KL = init(X, n_neigh=-1)
    K_max = len(H) 
    print 'K_max', K_max, len(H)
    w[6].grid_forget()
    w[6] = Tk.Scale(master=root, from_=0, to=K_max-1, tickinterval= K_max / 2, length=str(4) + 'i', label = 'step k', resolution = 1, troughcolor = 'magenta4',orient=Tk.HORIZONTAL)
    w[6].set(0)
    w[6].grid(row = 4, column = 2 * V + 1, columnspan = 3)
    weights_computed = [None] * (K_max)
    true_weights = np.zeros((n,n))
    f[4].clear()
    set_title(f)
    for cluster_i in true_clusters:
    	for i in cluster_i:
    		for j in cluster_i:
    			true_weights[i,j] = 1.
    ax4 = f[4].add_subplot(111)
    #((ax1, ax2), (ax3, ax4)) = f.add_subplot(221)
    ax4.imshow(true_weights, cmap=plt.cm.gray, interpolation='nearest')
    canvas[4].show()
    return 0

root,V,S = 0,0,0

def application():
    global f,canvas, w, texts, K_max, root, V, S
    #mpl.use('svg')
    root = Tk.Tk()
    plot_height = 4
    plot_width = 4
    
    f0 = Figure(figsize=(plot_width, plot_height), facecolor="white")
    f1 = Figure(figsize=(plot_width, plot_height), facecolor="white")
    f2 = Figure(figsize=(plot_width, plot_height), facecolor="white")
    f3 = Figure(figsize=(plot_width, plot_height), facecolor="white")
    f4 = Figure(figsize=(plot_width, plot_height), facecolor="white")
    canvas0 = FigureCanvasTkAgg(f0, master=root)
    canvas1 = FigureCanvasTkAgg(f1, master=root)
    canvas2 = FigureCanvasTkAgg(f2, master=root)
    canvas3 = FigureCanvasTkAgg(f3, master=root)
    canvas4 = FigureCanvasTkAgg(f4, master=root)
    canvas = [canvas0,canvas1,canvas2,canvas3,canvas4]
    f = [f0,f1,f2,f3,f4]
    set_title(f)
    w1 = Tk.Scale(master=root, from_=-20., to=10., tickinterval= 10, length=str(plot_height) + 'i', label = 'Lambda', resolution = 0.01, troughcolor = 'magenta4',orient=Tk.HORIZONTAL)
    w1.set(15)
    w2 = Tk.Scale(master=root, from_=0.0, to=8., tickinterval= 1, length=str(plot_height) + 'i', label = 'distance', resolution = 0.05, troughcolor = 'magenta4',orient=Tk.HORIZONTAL)
    w2.set(4.5)
    w3 = Tk.Scale(master=root, from_=500, to=0, tickinterval= 250, length=str(3) + 'i', label = 'N_1', resolution = 1, troughcolor = 'magenta4')#,orient=Tk.HORIZONTAL)
    w3.set(100)
    w4 = Tk.Scale(master=root, from_=500, to=0, tickinterval= 250, length=str(3) + 'i', label = 'N_2', resolution = 1, troughcolor = 'magenta4')#,orient=Tk.HORIZONTAL)
    w4.set(100)
    w5 = Tk.Scale(master=root, from_=500, to=0, tickinterval= 250, length=str(3) + 'i', label = 'N_3', resolution = 1, troughcolor = 'magenta4')#,orient=Tk.HORIZONTAL)
    w5.set(100)
    w6 = Tk.Scale(master=root, from_=0, to=K_max, tickinterval= 1, length=str(4) + 'i', label = 'step k', resolution = 1, troughcolor = 'magenta4',orient=Tk.HORIZONTAL)
    w6.set(0)
    
    root.wm_title("AWC illustration")
    
    w = [0,w1,w2,w3,w4,w5, w6]
    
    T = Tk.Text(root, height=1, width=10, font = 2)
    T.insert(Tk.END, "p")
    
    button1 = Tk.Button(master=root, text='Quit', width = 10,command=sys.exit)
    show_step = Tk.IntVar()
    show_movie = Tk.IntVar()
    clustering = Tk.IntVar()
    clustering.set(1) 
    #button.pack()
    button3 = Tk.Checkbutton(master=root, text="Step-by-step", variable=show_step)
    button2 = Tk.Button(master=root, text='Show',foreground = 'magenta4',height=1, width = 10, bd = 10, bg = 'magenta4', command=lambda: refresh(f, canvas,show_step, w, clustering,texts,show_movie))
    button4 = Tk.Button(master=root, text='2 Norm data', width = 10, bd = 10, bg = 'magenta4', command=lambda: get_2_normal(f, canvas,w))
    button5 = Tk.Button(master=root, text='3 Norm data', width = 10, bd = 10, bg = 'magenta4', command=lambda: get_3_normal(f, canvas,w))
    button6 = Tk.Checkbutton(master=root, text="Clustering", variable=clustering)
    button8 = Tk.Checkbutton(master=root, text="movie", variable=show_movie)
    button10 = Tk.Button(master=root, text='iris (d=4)', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/iris.txt', 0.5, True))
    button11 = Tk.Button(master=root, text='wine (d=13)', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/wine.dat', -17, True))
    button12 = Tk.Button(master=root, text='thy (d=5)', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/thy.arff', 0.3, True))
    button13 = Tk.Button(master=root, text='seeds (d=7)', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/seeds_dataset.txt', -2, True))
    button14 = Tk.Button(master=root, text='compound', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/compound.arff', 3.7, False))
    #button15 = Tk.Button(master=root, text='flame', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, 'flame.arff'))
    button15 = Tk.Button(master=root, text='orange', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/orange2.txt', 3, False))
    button16 = Tk.Button(master=root, text='aggregation', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/aggregation.arff', 3.5, False))
    button17 = Tk.Button(master=root, text='ds2c2sc13', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/ds2c2sc13.arff', 3.5, False))
    
    button18 = Tk.Button(master=root, text='pathbased', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/pathbased.arff', 4, False))
    button19 = Tk.Button(master=root, text='flame', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/flame.arff', 3.5, False))
    
    button20 = Tk.Button(master=root, text='ds4c2sc8', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/ds4c2sc8.arff', 4, False))
    button21 = Tk.Button(master=root, text='zelnik4', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '/dataset/zelnik4.arff', 6, False))
    #button22 = Tk.Button(master=root, text='dpc', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, 'dpc.arff'))
    #button23 = Tk.Button(master=root, text='3-spiral', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '3-spiral.arff'))
    #button24 = Tk.Button(master=root, text='orange1', width = 10,bd = 10, bg = 'magenta4', command=lambda: load_dataset_app(f, canvas,w, '../Data/orange2.txt'))
    
    
    
    
    
    
    error_1 = Tk.Text(master=root, width = 10, height = 1, background = 'lightgrey')
    error_1.insert(Tk.INSERT, "0.0")
    error_2 = Tk.Text(master=root, width = 10, height = 1, background = 'lightgrey')
    error_2.insert(Tk.INSERT, "0.0")
    error_1_l = Tk.Label(master=root, text=" error \n union", width = 10, height = 2)
    error_2_l = Tk.Label(master=root, text=" error \n propagation", width = 10, height = 2)
    
    S1_l = Tk.Label(master=root, text="Var 1", width = 5, height = 1)
    S2_l = Tk.Label(master=root, text="Var 2", width = 5, height = 1)
    S3_l = Tk.Label(master=root, text="Var 3", width = 5, height = 1)
    
    S1_11 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S1_12 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S1_21 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S1_22 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S2_11 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S2_12 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S2_21 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S2_22 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S3_11 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S3_12 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S3_21 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    S3_22 = Tk.Text(master=root, width = 3, height = 1, background = 'lightgrey')
    
    S1_11.insert(Tk.INSERT, "1")
    S2_11.insert(Tk.INSERT, "1")
    S3_11.insert(Tk.INSERT, "1")
    S1_22.insert(Tk.INSERT, "1")
    S2_22.insert(Tk.INSERT, "1")
    S3_22.insert(Tk.INSERT, "1")
    S1_12.insert(Tk.INSERT, "0")
    S2_12.insert(Tk.INSERT, "0")
    S3_12.insert(Tk.INSERT, "0")
    S1_21.insert(Tk.INSERT, "0")
    S2_21.insert(Tk.INSERT, "0")
    S3_21.insert(Tk.INSERT, "0")
    
    S1 = [S1_11,S1_12,S1_21,S1_22]
    S2 = [S2_11,S2_12,S2_21,S2_22]
    S3 = [S3_11,S3_12,S3_21,S3_22]
    S = [0, S1, S2, S3]
    
    
    
    texts = [0, error_1, error_2]
    
    V = 7
    w6.grid(row = 4, column = 2 * V + 1, columnspan = 3)
    w1.grid(row = 5, column = 2 * V + 1, columnspan = 3)
    w3.grid(row = 6, column = 2 * V + 1, rowspan = 3)
    w4.grid(row = 6, column = 2 * V + 2, rowspan = 3)
    w5.grid(row = 6, column = 2 * V + 3, rowspan = 3)
    w2.grid(row = 9 , column = 2 * V + 1, columnspan = 3)
    canvas0.get_tk_widget().grid(row = 0, column = 0, rowspan = V, columnspan = V)
    canvas1.get_tk_widget().grid(row = 0, column = V, rowspan = V, columnspan = V)
    canvas2.get_tk_widget().grid(row = V, column = 0, rowspan = V, columnspan = V)
    canvas3.get_tk_widget().grid(row = V, column = V, rowspan = V, columnspan = V)
    canvas4.get_tk_widget().grid(row = 0, column = 2 * V + 4, rowspan = V, columnspan = V)
    button1.grid(row = 2 * V-2, column = 2 * V + 2)
    button2.grid(row = 1, column = 2 * V + 1)
    button3.grid(row = 2, column = 2 * V + 1)
    button4.grid(row = 2, column = 2 * V + 2)
    button5.grid(row = 2, column = 2 * V + 3)
    button6.grid(row = 3, column = 2 * V + 1)
    #button7.grid(row = 3, column = 2 * V + 2)
    button8.grid(row = 0, column = 2 * V + 1)
    #button9.grid(row = 3, column = 2 * V + 3)
    button10.grid(row = V, column = 2 * V + 4)
    button11.grid(row = V, column = 2 * V + 5)
    button12.grid(row = V, column = 2 * V + 6)
    button13.grid(row = V + 1, column = 2 * V + 4)
    button14.grid(row = V + 1, column = 2 * V + 5)
    button15.grid(row = V + 1, column = 2 * V + 6)
    button16.grid(row = V + 2, column = 2 * V + 4)
    button17.grid(row = V + 2, column = 2 * V + 5)
    button18.grid(row = V + 2, column = 2 * V + 6)
    button19.grid(row = V + 3, column = 2 * V + 4)
    button20.grid(row = V + 3, column = 2 * V + 5)
    button21.grid(row = V + 3, column = 2 * V + 6)
    #button22.grid(row = V + 4, column = 2 * V + 4)
    #button23.grid(row = V + 4, column = 2 * V + 5)
    #button24.grid(row = V + 4, column = 2 * V + 6)
    
    
    
    
    
    error_1.grid(row = 1, column = 2 * V + 2)
    error_2.grid(row = 1, column = 2 * V + 3)
    error_1_l.grid(row = 0, column = 2 * V + 2)
    error_2_l.grid(row = 0, column = 2 * V + 3)
    
    '''
    S1_l.grid(row = V + 2 + 1, column = 2 * V + 4, columnspan = 1,rowspan = 1)
    S2_l.grid(row = V + 2 + 1, column = 2 * V + 6, columnspan = 1,rowspan = 1)
    S3_l.grid(row = V + 2 + 1, column = 2 * V + 8, columnspan = 1,rowspan = 1)
    
    S1_11.grid(row = V + 3 + 1, column = 2 * V + 4, columnspan = 1,rowspan = 1)
    S1_12.grid(row = V + 3 + 1, column = 2 * V + 5, columnspan = 1,rowspan = 1)
    S1_21.grid(row = V + 4 + 1, column = 2 * V + 4, columnspan = 1,rowspan = 1)
    S1_22.grid(row = V + 4 + 1, column = 2 * V + 5, columnspan = 1,rowspan = 1)
    
    S2_11.grid(row = V + 3 + 1, column = 2 * V + 6, columnspan = 1,rowspan = 1)
    S2_12.grid(row = V + 3 + 1, column = 2 * V + 7, columnspan = 1,rowspan = 1)
    S2_21.grid(row = V + 4 + 1, column = 2 * V + 6, columnspan = 1,rowspan = 1)
    S2_22.grid(row = V + 4 + 1, column = 2 * V + 7, columnspan = 1,rowspan = 1)
    
    S3_11.grid(row = V + 3 + 1, column = 2 * V + 8, columnspan = 1,rowspan = 1)
    S3_12.grid(row = V + 3 + 1, column = 2 * V + 9, columnspan = 1,rowspan = 1)
    S3_21.grid(row = V + 4 + 1, column = 2 * V + 8, columnspan = 1,rowspan = 1)
    S3_22.grid(row = V + 4 + 1, column = 2 * V + 9, columnspan = 1,rowspan = 1)
    '''
    
    
    root.lift()
    root.call('wm', 'attributes', '.', '-topmost', True)
    root.after_idle(root.call, 'wm', 'attributes', '.', '-topmost', False)
    Tk.mainloop()
    return 0


def get_seed_data(f, canvas, w):
	print os.path.dirname(os.path.abspath(__file__))
	global X, true_clusters
	X = np.genfromtxt(os.path.dirname(os.path.abspath(__file__)) + '/Data/seeds_dataset.txt')
	n = np.size(X,0)
	Y = X[:,7]
	true_weights = np.zeros((n, n))
	for i in range(n):
	    for j in range(n):
	        if Y[i] == Y[j]:
	            true_weights[i, j] = 1
	true_clusters = []
	for i in range(3):
	    true_clusters.append([])
	for i in xrange(len(Y)):
	    true_clusters[int(Y[i])-1].append(i) 
	n_components = 2
	faces = X[:, :-1]
	pca = decomposition.PCA(n_components=n_components)
	pca.fit(faces)
	X = pca.transform(faces)
	#X = faces
	#X = preprocessing.scale(X)
	f[3].clear()
	ax3 = f[3].add_subplot(111)
	for cluster in true_clusters:
		x = X[cluster, :]
		ax3.plot(x[:, 0], x[:, 1], 'o')
	ax3.axis('equal')
	set_title(f)
	canvas[3].show()
	get_number_of_steps()
	w[1].set(1.42)
	return 0
    
def get_orange_data():
    global X, true_clusters, f, canvas, X_PCA
    
    X = []
    r = np.linspace(4,5.5,4)
    dens = 10
    for r_i in r:
    	N_i = r_i * dens
    	points_i = np.linspace(0, 2 * math.pi * r_i, N_i) + np.random.uniform(0, 1, 1)
    	for point in points_i:
    		x = r_i * math.cos(point / r_i)
    		y = r_i * math.sin(point / r_i)
    		if X == []:
    			print x, y
    			X = np.zeros((1,2))
    			X[0, 0], X[0, 1] = x, y
    		else:
    			X = np.concatenate((X, [[x, y]]), axis=0)
    
    N1 = np.size(X, 0)
    true_clusters = [range(N1)]
    			
    r = np.linspace(0.01,3.3,7)
    dens = 7
    for r_i in r:
    	N_i = r_i * dens
    	points_i = np.linspace(0, 2 * math.pi * r_i, N_i) + np.random.uniform(0, 1, 1)
    	for point in points_i:
    		x = r_i * math.cos(point / r_i)
    		y = r_i * math.sin(point / r_i)
    		if X == []:
    			print x, y
    			X = np.zeros((1,2))
    			X[0, 0], X[0, 1] = x, y
    		else:
    			X = np.concatenate((X, [[x, y]]), axis=0)
    true_clusters += [range(N1, np.size(X, 0), 1)]
    if np.size(X, 1) == 2:
        X_PCA = copy.deepcopy(X)
    else:
        n_components = 2
        pca = decomposition.PCA(n_components=n_components)
        pca.fit(X)
        X_PCA = pca.transform(X)
    f[3].clear()
    ax3 = f[3].add_subplot(111)
    for cluster in true_clusters:
    	x = X[cluster, :]
    	ax3.plot(x[:, 0], x[:, 1], 'o')
    ax3.axis('equal')
    set_title(f)
    canvas[3].show()
    get_number_of_steps()
    w[1].set(1.42)
    return 0
  
def convert_to_numbers(Y):
    clusters = dict()
    result = []
    n = 0
    for y in Y:
        if y in clusters.keys():
            result.append(clusters.get(y))
        else:
            clusters[y] = n
            result.append(n)
            n = n+1
    return result

X_PCA = 0

def load_dataset_app(f, canvas,w, filename, w1_value=4, normalize=True):
    global X, true_clusters, X_PCA, cluster_colors
    load_dataset(filename, normalize)
    if np.size(X, 1) == 2:
        X_PCA = copy.deepcopy(X)
    else:
        n_components = 2
        pca = decomposition.FastICA(n_components=n_components)
        pca.fit(X)
        X_PCA = pca.transform(X)
    
    
    #X = faces
    #X = preprocessing.scale(X)
    f[3].clear()
    ax3 = f[3].add_subplot(111)
    for cluster_i in range(len(true_clusters)):
        x = X_PCA[true_clusters[cluster_i], :]
        ax3.plot(x[:, 0], x[:, 1], 'o', color=cluster_colors[cluster_i])
    ax3.axis('equal')
    set_title(f)
    canvas[3].show()
    get_number_of_steps()
    w[1].set(w1_value)
    return 0
    
  
def load_dataset(filename, normalize = True):
    global X, true_clusters
    path = os.path.dirname(os.path.abspath(__file__))
    d = 0
    X = np.genfromtxt(path + filename)
    
    if filename == '../Data/yeast.txt':
        X = np.genfromtxt(path + filename)
        X = X[:, 1:]
        Y = np.genfromtxt(path + filename,  dtype='str')[:, -1]
        Y = convert_to_numbers(Y)
        Y = np.asarray(map(int, Y))
        X = X[:, :-1]
        X = np.delete(X, [4,5], 1)

    else:
        if np.isnan(X).any():
            X = np.genfromtxt(path + filename, delimiter=',')
            d = 1 
        Y = X[:,-1]
        if np.isnan(Y).any():
            if d == 0:
                Y = np.genfromtxt(path + filename,  dtype='str')[:, -1]
            else:
                Y = np.genfromtxt(path + filename, delimiter=',', dtype='str')[:, -1]
            Y = convert_to_numbers(Y)
        Y = np.asarray(map(int, Y))
        X = X[:, :-1]
    
    if filename == 't4,8k.dat':
        X = np.genfromtxt(path + '/artificial/' + filename)
        #X = X[X[:, 0] < 300, :]
        Y = np.ones((np.size(X, 0),))
        true_clusters = [range(np.size(X, 0))]
        return X, Y, true_clusters, filename
    
    #print np.shape(X)
    #plt.plot(X[:, 0], X[:, 1], 'or')
    #plt.show()
         
        
        
    #X = X[:, :-1]
    #X = X[:, :-1]
    #normalize
    if normalize:
        print 'rrrrererererr'
        for i in xrange(np.size(X, 1)):
            print i,  (np.sum(X[:, i] ** 2))
            X[:, i] /= (np.sum(X[:, i] ** 2)) ** 0.5
        
    
    X = X[Y.argsort()]
    Y = Y[Y.argsort()]
    n = np.size(X,0)
    true_clusters = []
    cluster_marker = []
    for i in xrange(n):
        if Y[i] in cluster_marker:
            true_clusters[cluster_marker.index(Y[i])].append(i)
        else:
            cluster_marker.append(Y[i])
            true_clusters.append([i])
    #for i in cluster_marker:
    #   plt.plot(X[Y==i, 0], X[Y==i, 1], 'o')
    
    f = filename.replace('.arff', '')
    f = f.replace('.txt', '')
    import re
    f = re.sub("\.\.\/.*\/", "", f)
    '''print f
    plt.savefig(path + '/original/' + f+ '.png')
    plt.clf()'''
    #plt.show()
    return X, Y, true_clusters, f

def main():
	application()
	return 0

if __name__ == '__main__':
	main()