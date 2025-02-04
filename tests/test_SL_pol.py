import mixedpickinginventorymodel.PolicyEvaluation.PolicyEvaluationSL as evaluate
import numpy as np
import pandas as pd
import pickle
import sys
import os
if __name__ == '__main__':
    instance_2_run = int(sys.argv[3])
    to_eval = ['negbin-5-10','negbin-5-15','negbin-10-10','negbin-10-15', 'pois-5-10', 'pois-5-15', 'pois-10-10', 'pois-10-15']
    
    np.random.seed(37)
    # Open policy
    directory = "/beegfs/client/default/loweryb/sustainability/instances/{}/".format(to_eval[instance_2_run])
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            pol = pickle.load(open(f,'rb'))
            optimal_policy = pd.DataFrame(pol['optimal_pol'], columns=['period','Inventory','Order'])
            print("Testing policy {} with ".format(pol['fifo_rate']))
            if instance_2_run <= 3:
                instance=evaluate.PolicyEvaluationSL(pol['T'], pol['m'], pol['h'], pol['p'], pol['c'], pol['theta'], pol['psi'], 0.999, ['NegBin', [6, 0.375]], 35,int(sys.argv[1]), float(sys.argv[2]))
            else:
                instance=evaluate.PolicyEvaluationSL(pol['T'], pol['m'], pol['h'], pol['p'], pol['c'], pol['theta'], pol['psi'], 0.999, ['Poisson', [10]], 25,int(sys.argv[1]), float(sys.argv[2]))
            instance.input_policy_rate = pol['fifo_rate']
            print(instance.num_cores)
            instance.terminal_state()
            instance.EvalPolicy(optimal_policy)