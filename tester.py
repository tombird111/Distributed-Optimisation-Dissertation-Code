import dill as pickle
import numpy as np
import sys
from mpi4py import MPI
from disropt.agents import Agent
from disropt.algorithms.primal_decomp import PrimalDecomposition
from disropt.functions import QuadraticForm, Variable
from disropt.utils.utilities import is_pos_def
from disropt.constraints.projection_sets import Box
from disropt.utils.graph_constructor import binomial_random_graph
from disropt.problems.constraint_coupled_problem import ConstraintCoupledProblem
from primal_update import UpdatePrimalDecomposition
from ga_primal import GAPrimalDecomposition

"""When using this python script, arguments should be appended afterwards, based on a number of properties

The script should be called with: mpiexec -np [i] python tester.py from the command line
The [i] within this should be replaced with a number, representing the number of nodes within the graph

A sequence of arguments should be added after the tester.py, appearing in the form:
mpiexec -np [i] python tester.py [arg1] [arg2] [arg3]...

[arg1] represents the name of the test, and is mainly used for choosing file location: for example "test1"
[arg2] represents the random seed to use on each node to ensure recreatable results. This would be a number, for example, 3
[arg3] represents the number of constraints to use, for example, 2
[arg4] represents the dimensions of the problem to use
[arg5] represents which solving algorithm to use
"""
if len(sys.argv) == 6:
    # Get MPI info
    # This is for use of mpi4py and running multiple separate processes
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    local_rank = comm.Get_rank()

    # Generate a common graph (everyone use the same seed)
    Adj = binomial_random_graph(nproc, p=0.3, seed=1)

    # Reset local seed
    np.random.seed(int(sys.argv[2]))

    # Create an agent for the current program. This will be based on the MPI process
    agent = Agent(
        in_neighbors=np.nonzero(Adj[local_rank, :])[0].tolist(),
        out_neighbors=np.nonzero(Adj[:, local_rank])[0].tolist())

    # Local variable dimension
    # This can be increased/decreased to simplify tests
    n_i = int(sys.argv[4])

    # Number of coupling constraints
    S = int(sys.argv[3])

    # Generate a positive definite matrix
    P = np.random.randn(n_i, n_i)
    P = P.transpose() @ P
    bias = np.random.randn(n_i, 1)

    # Declare a variable
    x = Variable(n_i)

    # Define the local objective function
    fn = QuadraticForm(x - bias, P)

    # Define the local constraint set, based on the number of dimensions
    low = -2*np.ones((n_i, 1))
    up = 2*np.ones((n_i, 1))
    constr = Box(low, up)


    # Define the local contribution to the coupling constraints (based on dimensions and constraints)
    A = np.random.randn(S, n_i)
    coupling_fn = A.transpose() @ x
    
    # Create local problem and assign to agent
    # Problem is based on the previously created objective functions and constraints
    pb = ConstraintCoupledProblem(objective_function=fn,
                                  constraints=constr,
                                  coupling_function=coupling_fn)
    agent.set_problem(pb)

    # Initialize allocation
    y0 = np.zeros((S, 1))

    # Create the algorithm
    
    algo_arg = sys.argv[5]
    if algo_arg == "ga":
        algorithm = GAPrimalDecomposition(agent=agent, initial_condition=y0, enable_log=True, ga_seed = int(sys.argv[2]))
    elif algo_arg == "uninfga":
        algorithm = GAPrimalDecomposition(agent=agent, initial_condition=y0, enable_log=True, smart_init = False, ga_seed = int(sys.argv[2]))
    else:
        algorithm = UpdatePrimalDecomposition(agent=agent, initial_condition=y0, enable_log=True)

    def step_gen(k): # Define a stepsize generator
        return 0.1/np.sqrt(k+1)

    # Run the algorithm
    x_sequence, y_sequence, J_sequence = algorithm.run(iterations=100, stepsize=step_gen, M=100.0, verbose=True)
    x_t, y_t, J_t = algorithm.get_result()
    print("Agent {}: primal {} allocation {}".format(agent.id, x_t.flatten(), y_t.flatten()))

    test_name = str("Tests/" + sys.argv[1] + "/")
    agent_file_name = str(test_name + "agents.npy")

    np.save(agent_file_name, nproc)

    # save agent and sequence
    with open(str(test_name + 'agent_{}_obj_function.pkl'.format(agent.id)), 'wb') as output:
        pickle.dump(agent.problem.objective_function, output, pickle.HIGHEST_PROTOCOL)
    with open(str(test_name + 'agent_{}_coup_function.pkl'.format(agent.id)), 'wb') as output:
        pickle.dump(agent.problem.coupling_function, output, pickle.HIGHEST_PROTOCOL)
    np.save(str(test_name + "agent_{}_allocation_sequence.npy".format(agent.id)), y_sequence)
    np.save(str(test_name + "agent_{}_primal_sequence.npy".format(agent.id)), x_sequence)
else:
    print("Insufficient number of arguments called within the command line")