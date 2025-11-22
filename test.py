# Example of loading a dataset
from load_dataset import load_vrp_dataset

# Load a CVRP dataset with 50 customers, single depot configuration
dataset = load_vrp_dataset('cvrp', 50, 'single_depot')

# Run evaluation
from vrp_bench.vrp_evaluation import VRPEvaluator

evaluator = VRPEvaluator()
results = evaluator.evaluate_solver(solver_class=ACOSolver, 
                                   solver_name="ACO",
                                   sizes=[50, 100])