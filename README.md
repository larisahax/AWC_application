# AWC
Adaptive Weights Clustering

This code shows the performance of the AWC algorithm on several data sets.

Requirements:
python version 2.7

libraries:
matplotlib
scipy
igraph
math
numpy
pylab
sklearn
Tkinter


Run the AWC application: python AWC_app.py

Data generation

One can upload data examples by pressing one of the buttons on the right (the optimal \lambda will be set automatically)

1)Real World Data:
iris, wine, thy, seeds

d is data dimension

The windows show a two-dimensional projection of data using ICA, but the algorithm runs with the initial dimension.
 
2)Artificial data:
compound, orange, aggregation, ds2c2sc13, pathbased, flame, ds4c2sc8, zelnik4

These are two dimensional data taken from https://github.com/deric/clustering-benchmark

3) Gaussian data
One can test AWC also on data from Gaussian distribution.
button (2 Norm data)  will generate 2 clusters from Normal distribution
N_1 points from N((0,0), Var 1)
N_2 points from N((distance,0), Var 2)

button (3 Norm data) will generate 3 clusters with (distance) between means
N_1 points from N(, Var 1)
N_2 points from N(, Var 2)
N_3 points from N(, Var 3)

Matrices (Var 1, Var 2, Var 3), (N_1,N_2,N_3) and (distance) are flexible.

Launch Examples:

After selecting data AWC uploads the data and is ready for launch.

To see the final result of AWC press (Show)

It is possible to see the clustering process for each point.
Select a point in the window (AWC weights for one point), select  (movie) and then (Show)
The results for the selected point will be shown in dynamics.

Also it is possible to see the result of each step
For that choose (step-by-step) and (Show)
Only one step will be taken each time you press

To see the results for step k, choose the step (step k) and select (step-by-step)

If one wants to see the dynamics for a selected point started from step k, choose (step k), select a point in (AWC weights for one point), then select (movie) and press (Show).

 Windows:

On the left there are 4 windows
(AWC weights) - the weight matrix at the current step
(AWC weights for one point) - weight values of the selected point with the rest colored from red to white (from high to low)
(AWC clustering) - clustering based on weight matrix (plotting is heavy)
(true/wanted clustering) - true (for ready data) and right (for normal Cluster) Clustering
Top Right window 
(True / wanted weights) - true (for ready data) and the right (for normal cluster) weight matrix

Launch AWC

1)The (Lambda) parameter is set as recommended for given data sample, but it can be changed as desire

2)(Show) shows the final result after the last step.

3)If (clustering) is selected then the clustering based on the weights matrix will be shown in the window (AWC clustering) 

4)If you are interested in specific step, it’s possible to select step k.
Also one can select a point and see it’s weights in the window (AWC weights for one point) 

5) By selecting (movie) and pressing (Show) the changes in all windows will be shown non-stop started from the selected step (1 sec per step).
So it is possible to observe how connections of a point spread gradually.

6)If choose (step-by-step) without (movie), then after pressing (Show) the results of the step k+1 will be shown.

7)Also errors (error union) and (error propagation) are shown
(error union) - counts all connections (positive weights) between points from different clusters
(error propagation) -  indicates the number of disconnecting points in the same cluster

The button (Quit) quits the application.



