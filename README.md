# AWC application
Adaptive Weights Clustering

Authors: K. Efimov, L. Adamyan, V. Spokoiny

This application demonstrates the performance of the AWC algorithm on several data sets.

Requirements:
python version 2.7

libraries:

matplotlib;
scipy;
math;
numpy;
pylab;
sklearn;
Tkinter


Run the AWC application: *python AWC_application.py*

First of all you need to load the data.

**Data generation**

One can upload data examples by pressing one of the buttons on the right (the optimal \lambda will be set automatically)

1) Real World Data: 
*iris, wine, thy, seeds*

These data sets are taken from the UCI repository: https://archive.ics.uci.edu/ml/datasets.html

*d* is data dimension

The windows on the left show a two-dimensional projection of data using ICA, but the algorithm works with the initial dimension.
 
2) Artificial data:
*compound, orange, aggregation, ds2c2sc13, pathbased, flame, ds4c2sc8, zelnik4*

These are two dimensional data sets taken from https://github.com/deric/clustering-benchmark

3) Gaussian data

One can test AWC also on data from Gaussian distribution.
Button (2 Norm data)  will generate 2 clusters from Normal distribution:
N_1 points from N((0,0), Identity)
N_2 points from N((distance,0), Identity)

Button (3 Norm data) will generate 3 normal clusters with (distance) between means and identtiy variance:
N_1, N_2, N_3 points in the first, second and third cluster correspondingly.

(N_1,N_2,N_3) and (distance) are flexible.

**Launch Examples:**

After selecting data AWC uploads the data and is ready for launch.

To see the final result of AWC press (Show).

It is possible to see the result of each step.
For that choose (step-by-step) and push (Show)
Only one step will be taken each time you press (Show).

To see the results for step k, choose the step (step k) and select (step-by-step)

Also it is possible to see the clustering process for each point.
Select a point in the window (AWC weights for one point), select  (movie) and then (Show).
The results for the selected point will be shown in dynamics.
(the regime (movie) perhaps will work slowly because of slow drawing, not the AWC)

Moreover if one wants to see the dynamics for a selected point started from step k, choose (step k), select a point in (AWC weights for one point), then select (movie) and press (Show).

**Windows:**

On the left there are 4 windows
(AWC weights) - the weight matrix at the current step (white means 1, black is 0)

(AWC weights for one point) - shows the weights values between the selected point and the rest. The higher the weight, darker the color (from red to white - from 1 to 0). In other words this window shows one raw of the weight matrix at the current step

(AWC clustering) - clustering based on weight matrix

(true/wanted clustering) - true (for ready data) and optimal (for normal cluster) clustering.

Top Right window 

(True / wanted weights) - true (for ready data) and the optimal (for normal cluster) weight matrix.

**Launch AWC**

1) The (Lambda) parameter is set as recommended for given data sample, but it can be changed as desired.

2) (Show) shows the final result after the last step.

3) If (clustering) is selected then the clustering based on the weights matrix will be shown in the window (AWC clustering) 

4) If you are interested in specific step, it’s possible to select step k.
Also one can select a point and see it’s weights in the window (AWC weights for one point) 

5) By selecting (movie) and pressing (Show) the changes in all windows will be shown non-stop started from the selected step (1 sec per step).
So it is possible to observe how connections of a point spread gradually.

6) By selecting (step-by-step) without (movie), and pressing (Show) the results of the step k+1 will be shown.

7) Also errors (error union) and (error propagation) are shown

(error union) - counts all connections (positive weights) between points from different clusters

(error propagation) -  indicates the number of disconnecting points in the same cluster

The errors are normalized to 0-1.

8) The button (Quit) closes the application.



