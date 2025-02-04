"""
    Model of Clarkson et al. (2022) Version 5 (Storm edition)

    Solve optimal policy.

    * Changes to V5: 
        1. Parrellised
        2. Built for Storm 

"""

import numpy as np
import scipy.stats as sp
import mixedpickinginventorymodel.binom_cython as binom_cython
import itertools
import pickle
import uuid
import multiprocessing as mp
import math


# Optimal perishing class
class optimal_perishing():
    """
    Simulates an inventory system over a specified number of periods, based on the 
    parameters specified during initialization. The system tracks inventory levels, 
    demand, order quantities, and perishing of a product.

    Parameters:
    -----------
    periods : int
        The number of periods to simulate.
    lifetime : float
        The penalty cost for each unit out of stock.
    holding_c : float
        The holding cost for each unit of inventory still in stock.
    penalty_c : float
        The cost for each unit of perished inventory.
    order_c : float
        The order cost for each unit
    outdating_c : float
        The cost per unit of outdated items
    perish_probs : List(float)
        The perish rates in each age class, where the i-th element represents the 
        perish rate of the i-th class.
    discount_factor : float
        The gamma parameter for discounting future costs in the Dynamic Program
    demand_params : List(string, List(float))
        The distribution and respective demand parameters 
    demand_truncation : int
        The maximum demand (and subsequently order quantity) that can be taken.
    FIFO_rate : float
        The percentage of demand that is experienced as FIFO, the remaining 1-rate is the LIFO choice

    Example:
    --------
    if __name__ == '__main__':

        # Instance Structure

        # lifetime
        m=2

        # Time horizon
        T=10

        # holding cost
        h=1

        # penalty cost
        p=5

        # outdating cost
        theta=15
        
        # order cost
        c=5
        
        # perish probability (for the m-1 classes)
        psi = [0.3]


        # Discount Factor
        gamma=0.999

        # Demand params (negative binomial)
        nb_1 = 5
        nb_2 = 0.2 # Negative binomial is defined differently in paper compared to scipy (in scipy its 1-p for parameter)


        # Look at policies from 0 to 1 for FIFO rate in stages of 10%
        for i in range(11):
            print('FIFO rate: {}'.format(i),end = '\n')
            instance=optimal_pol.optimal_perishing(T, m, h, p, c, theta, psi, gamma, ['NegBin', [5,0.2]], 50,4,i*0.1)
            instance.terminal_state()
            instance.run_dp_algo()
    """
    def __init__(self, periods, lifetime, holding_c, penalty_c, order_c, outdating_c, perish_prob, discount_factor, demand_params, demand_truncation, num_cores, FIFO_rate=1):
        
        # Assign inputs
        self.T = periods
        self.m = lifetime
        self.h = holding_c
        self.p = penalty_c
        self.c = order_c
        self.theta = outdating_c
        self.psi = perish_prob
        self.gamma = discount_factor
        self.params = demand_params # Assumed stationary, can be poisson or negative binomial atm
        self.max_d = demand_truncation
        self.demand_val, self.demand_pmf = self.gen_demand()
        self.fifo_rate = FIFO_rate # Rate at which we assume customers are fifo/lifo in the simulation, used for the calc_post_demand() function

        self.max_q = self.max_d # Maximum order is essentially capped by maximum demand possible (says in jakes code: V3_four_ages_doc_heurs_4_Ex_fns_choose_maslv_ordcost.jl line 58)

        self.save_output = True # Assume we can save output to pickle file, might want to set to false if running small time tests but need to do so manually


        # Save the number of cpus to use when parallising
        self.num_cores = num_cores


        # Store states and answers
        self.optimal_pol = []
        self.V = [] # Previous states

        ########################
        # Pre-processing steps #
        ########################

        # Generate the state space
        self.state_space = [x for x in itertools.product(*[[i for i in range(self.max_q)] for j in range(self.m)])]

        # Pre-calculate the cost functions, parellised
        self.G = {}
        
        print('Pre-calculating cost function... ', end=' ')
        
        cl=mp.Pool(num_cores)
        # Map the calculation of the cost function. State space ar eall independent so can be computed in parallel
        res = cl.map(self.calc_immediate_cost,self.state_space)
        x_max = res
        # Save to dictionary to call upon later
        for idx, x in enumerate(self.state_space):
            self.G[x] = (res[idx])
        # Close the process.
        cl.close()
        
        print('Done! State space size: {} \n'.format(len(self.state_space)))



    def gen_demand(self):
        # Check which distribution we are checking
        if self.params[0] == 'Poisson':
            distr = sp.poisson(self.params[1][0])

        elif self.params[0] == 'NegBin':     
            # Generate new random negative binomial 
            distr = sp.nbinom(self.params[1][0], self.params[1][1])
        
        # Generate value/pmf pairings
        vals = [i for i in range(self.max_d)]
        pmf = [distr.pmf(v) for v in vals]
        return vals, pmf
    
    
    def calc_post_demand(self,x,d):
        """
            Calculates the post demand state according to LIFO and FIFO as a function of demand
            i.e. in Clarkson (2022) its the y_{t,i}(d_t)
        """

        y = []

        # Get split
        FIFO_d = round(self.fifo_rate*d)
        LIFO_d = d-FIFO_d
        
        # Realise FIFO
        for i in range(self.m):
            x_rest = sum([x[j] for j in range(i+1,self.m)])
            y.append(max(x[i]-max(FIFO_d-x_rest,0),0))
        
        # Realise LIFO
        for i in range(1,self.m+1):
            y_rest = sum([x[j] for j in range(i-1)])
            y[i-1] = max(y[i-1]-max(LIFO_d-y_rest,0),0)
        
        return y

    def terminal_state(self):
        """
            Appends the terminal condition for the SDP - V_{T+1} in notation, see Eq. 8 in Clarkson et al.(2022)
        """
        # Calculate mean demand
        mean_demand = np.sum([d*v for d,v in zip(self.demand_val, self.demand_pmf)])

        terminal_sum = np.sum([self.gamma**(t-2)*mean_demand for t in range(self.T)]) # Calculate sum for terminal cost
        V_T_plus_1 = (self.gamma**(-self.T)*self.c*terminal_sum) # calculate full terminal cost
        # Assign terminal cost to each possible state
        for x in self.state_space:
            self.V.append((list(x), V_T_plus_1))
        return

    def calc_immediate_cost(self, x):
        """
            Calculate the immediate cost function G(x_t)
        """

        # Keep track of expectation
        Exp = 0
        
        x_tot = sum(x) # get total on hand inventory

        # Step 1. Discretise expectation for the parts depending on the demand
        for (d,d_pmf) in np.dstack((self.demand_val, self.demand_pmf))[0]:
            Exp += d_pmf*(self.h*max(x_tot-d,0)+self.p*max(d-x_tot,0))
        
            # Step 2. Discretise the expectation for the outdating units, sum of binomial r.v
            y_d = self.calc_post_demand(x,d) # Calculate post demand state
            E_t = 0 # Log outdating costs
            for i in range(self.m-1): # Don't loop over the last demand class since we know this perishes with probability 1 
                # Calculate mean of binomial for outdating. just n*p.
                E_t += (y_d[i]*self.psi[i])

            # Append last demand class
            E_t += y_d[-1]

            Exp += d_pmf*self.theta*E_t # Add all outdating costs to expectation calculation
        return Exp
    
    def calc_perish_combos(self, x, d):
        """
            Calculate the j perishing combinations set, and returns the post demand state too
        """
        y_t = self.calc_post_demand(x,d) # Calculate post demand state
        
        # generate set, ignore the last demand class, might be a bit messy with list comprehension but seemed most intuitive way
        j_d = [j for j in itertools.product(*[[j_i for j_i in range(int(y_i)+1)] for y_i in y_t[:-1]])]

        return y_t, j_d
    
    def calc_future_cost(self,x, q, V_t_plus_1):
        fut_cost = 0
        for d in range(self.max_d):
            # Calculate the set of inventory combinations after perishing, and also keep post demand state for use in pmf calculation
            y_t, j_d = self.calc_perish_combos(x,d)
            inner_fut_cost = 0
            for j in j_d: # Iterate through perishing combinations
                
                binom_perish_f = []
                for i in range(self.m-1): # each demand class

                    binom_perish_f.append(binom_cython.binom_cython(j[i], y_t[i], 1-self.psi[i])) # call cython function to calculate pmf
                    
                # Multiply by future value function
                func_val_prod = binom_perish_f[0]*binom_perish_f[1]*V_t_plus_1[(q,)+j]
                inner_fut_cost += func_val_prod
            # Multiply inner cost function by the demand pmf
            fut_cost += inner_fut_cost*self.demand_pmf[d]
        return fut_cost

    def calculate_cost_to_go(self, x, V_t_plus_1,t):
        """
            Calculate the J_t(x,q) function
        """
        total_cost = {}

        im_cost = self.G[x]

        for q in range(self.max_q):
            fut_cost = self.gamma*self.calc_future_cost(x,q,V_t_plus_1)
            total_cost[q] = im_cost+fut_cost
        opt = min(total_cost, key=total_cost.get)
        return [x, opt, total_cost[opt]]

    def run_dp_algo(self):
        
        # Step 1. Iterate backwards recursively through all periods
        for period in range(self.T,0,-1):
            # Keep  one step ahead optimal cost
            V_t_plus_1 = {tuple(val_func[0]):val_func[1] for val_func in self.V}
            self.V = []
            # Step 2. Enumerate through all the possible inventory levels
            # Parellelised
            print('Period: {}...'.format(period))
            cl = mp.Pool(self.num_cores)
            # Use starmap as we have multiple function arguments
            x_q_v_triplets = cl.starmap(self.calculate_cost_to_go, list(zip(self.state_space,itertools.repeat(V_t_plus_1),itertools.repeat(period))))
            cl.close()

            # Save optimal policy and V_{t+1} function
            for x,q,v in x_q_v_triplets:
                self.V.append((x,v))
                self.optimal_pol.append((period, x,q))
            print('Done!')

        # Save object to file
        if(self.save_output):
            ext = str(uuid.uuid4()) # Generate unique string to not have overwriting objects
            file = open('/beegfs/client/default/loweryb/sustainability/instances/Optimal_policy_instance_m_'+str(self.m) + '_fifo_'+ str(self.fifo_rate) +'_' + ext  + '.pkl','wb')
            file.write(pickle.dumps(self.__dict__))
            file.close()

            print('Object saved w/ extension: ' +  ext)
