# Mixed Picking Inventory Model

Code for paper.

* The Dynamic Program can be found in `src/mixedpickinginventorymodel/OptimalPolicy`
* Policy evaluation for cost/waste/alpha service level is found in `src/mixedpickinginventorymodel/PolicyEvaluation`
* Code used to run paper results in found in `tests/*`
* Note the cut off for Poisson and Negative Binomial need to be changed based on the mean used (i.e. you want to take any demand realisation up to the .999 percentile)
 