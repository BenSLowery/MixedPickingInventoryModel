import mixedpickinginventorymodel.PolicyEvaluation.PolicyEvaluationRandomPicking as eval
import numpy as np
import pandas as pd
import pickle
import sys
import os
if __name__ == '__main__':
    np.random.seed(37)
    # Open policy
    directory = "./instances/"
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            pol = pickle.load(open(f,'rb'))
            optimal_policy = pd.DataFrame(pol['optimal_pol'], columns=['period','Inventory','Order'])

            instance=eval.PolicyEvaluation(pol['T'], pol['m'], pol['h'], pol['p'], pol['c'], pol['theta'], pol['psi'], 0.999, ['NegBin', [6,0.375]], 25,int(sys.argv[1]))
            instance.input_policy_rate = pol['fifo_rate']
            instance.terminal_state()
            instance.EvalPolicy(optimal_policy)