import perishable_optimal as optimal_pol
import sys
import time

if __name__ == '__main__':

    # Instance Structure

    # lifetime
    m=3

    # Time horizon
    T=22

    # holding cost
    h=1

    # penalty cost
    p=10

    # outdating cost
    theta=10

    # order cost
    c=0

    # perish probability (for the m-1 classes)
    psi = [0.1,0.2]


    # Discount Factor
    gamma=0.999

    # Number of parallel processes to use
    num_cores = int(sys.argv[1])

    #FIFO rate
    f_r = float(sys.argv[2])
    # Demand params (negative binomial)
    nb_1 = 6
    nb_2 = 0.375 # Negative binomial is defined differently in paper compared to scipy (in scipy its 1-p for parameter)
    # Demand Params (poisson)
    lam = 10

    #time results
    s = time.time()
    instance=optimal_pol.optimal_perishing(T, m, h, p, c, theta, psi, gamma, ['NegBin', [nb_1,nb_2]], 25,num_cores,f_r)
    instance.save_output = True
    instance.terminal_state()
    print('Terminal States added')
    instance.run_dp_algo()
    e=time.time()
    print("Time to run: {}".format(e-s))
