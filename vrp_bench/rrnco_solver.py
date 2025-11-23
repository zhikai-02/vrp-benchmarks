import sys
import os
import torch
import numpy as np
import time
from tensordict import TensorDict
from rl4co.envs.common.base import RL4COEnvBase

# Monkey patch RL4COEnvBase.__getstate__ to handle RNG state saving issues
def patched_getstate(self):
    state = self.__dict__.copy()
    if hasattr(self, "rng"):
        if isinstance(self.rng, torch.Tensor):
             state["rng"] = self.rng
        elif hasattr(self.rng, "get_state"):
             state["rng"] = self.rng.get_state()
    return state

RL4COEnvBase.__getstate__ = patched_getstate

# Monkey patch RL4COEnvBase.__setstate__ to handle RNG state loading issues
def patched_setstate(self, state):
    # Standard unpickling
    self.__dict__.update(state)
    
    # Ensure rng exists
    if not hasattr(self, "rng"):
        # Create a new generator if missing
        self.rng = torch.Generator()
        
    try:
        if "rng" in state:
            self.rng.set_state(state["rng"])
    except (TypeError, RuntimeError) as e:
        # print(f"Warning: Failed to restore RNG state. Ignoring.")
        pass
    except Exception as e:
        # print(f"Warning: Unexpected error restoring RNG state: {e}")
        pass

RL4COEnvBase.__setstate__ = patched_setstate

# Add parent directory to path to import rrnco
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vrp_base import VRPSolverBase
from rrnco.models import RRNet
from rrnco.envs.rcvrp import RCVRPEnv
# from rrnco.envs.rcvrptw import RCVRPTWEnv # Check if this exists

# Patch VRPContext to fix dimension mismatch
from rl4co.models.nn.env_embeddings.context import VRPContext, EnvContext

if not hasattr(VRPContext, "_patched"):
    # Patch _cur_node_embedding to handle batch_size=1 squeeze issue
    # We need to patch the method on the class that actually defines it or is used.
    # VRPContext inherits from EnvContext.
    
    original_cur_node_embedding = EnvContext._cur_node_embedding
    original_state_embedding = VRPContext._state_embedding
    
    def patched_cur_node_embedding(self, embeddings, td):
        cur = original_cur_node_embedding(self, embeddings, td)
        # print(f"DEBUG: cur dim: {cur.dim()}, shape: {cur.shape}")
        # If cur is 1D, it means [E], missing batch dim? Or [B], missing embed dim?
        # Usually it should be [B, E].
        if cur.dim() == 1:
             # Assuming it squeezed the batch dim if B=1
             cur = cur.unsqueeze(0)
        return cur
    
    def patched_state_embedding(self, embeddings, td):
        state = original_state_embedding(self, embeddings, td)
        # print(f"DEBUG: state dim: {state.dim()}, shape: {state.shape}")
        if state.dim() == 1:
             state = state.unsqueeze(0)
        return state
    
    # Patching EnvContext because VRPContext likely uses it
    EnvContext._cur_node_embedding = patched_cur_node_embedding
    VRPContext._state_embedding = patched_state_embedding
    VRPContext._patched = True

class RRNCOSolver(VRPSolverBase):
    def __init__(self, data):
        super().__init__(data)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Paths to checkpoints
        self.ckpt_cvrp = "/home/zihan/zhikai/iclr/real-routing-nco/checkpoints/rcvrp/epoch_199.ckpt"
        self.ckpt_twcvrp = "/home/zihan/zhikai/iclr/real-routing-nco/checkpoints/rcvrptw/epoch_199.ckpt"
        
        # Load models lazily or upfront? Upfront is better to fail early.
        # But we might only run one type.
        self.models = {}
        self.envs = {}
        
    def _get_model_and_env(self, is_tw):
        key = "twcvrp" if is_tw else "cvrp"
        if key in self.models:
            return self.models[key], self.envs[key]
            
        ckpt_path = self.ckpt_twcvrp if is_tw else self.ckpt_cvrp
        
        if not os.path.exists(ckpt_path):
            print(f"Warning: Checkpoint not found at {ckpt_path}")
            return None, None
            
        print(f"Loading RRNCO model from {ckpt_path}...")
        
        # Monkey patch torch.load to force map_location because rl4co might call it without map_location
        original_load = torch.load
        def patched_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = self.device
            if "weights_only" not in kwargs:
                kwargs["weights_only"] = False
            return original_load(*args, **kwargs)
        
        torch.load = patched_load
        try:
            model = RRNet.load_from_checkpoint(ckpt_path, map_location=self.device)
        finally:
            torch.load = original_load
            
        model.eval()
        
        # Patch missing attributes for legacy checkpoints
        from rrnco.models.nn.attn_freenet import GatingNeuralAdaptiveBias
        
        for module in model.modules():
            if module.__class__.__name__ == "AttnFree_Block":
                if not hasattr(module, "nab_type"):
                    print("DEBUG: Patching missing nab_type in AttnFree_Block")
                    module.nab_type = "gating"
                
                if not hasattr(module, "neural_adaptive_bias"):
                    print("DEBUG: Patching missing neural_adaptive_bias in AttnFree_Block")
                    if hasattr(module, "embed_dim"):
                        module.neural_adaptive_bias = GatingNeuralAdaptiveBias(
                            embed_dim=module.embed_dim,
                            use_duration_matrix=False
                        ).to(self.device)
                    else:
                        print("ERROR: Cannot patch neural_adaptive_bias, embed_dim missing")
        
        if is_tw:
            # Import RMTVRPEnv here to avoid error if not exists
            from rrnco.envs.rmtvrp import RMTVRPEnv
            env = RMTVRPEnv(normalize=True)
        else:
            env = RCVRPEnv(normalize=True)
            
        print(f"DEBUG: env.rng type: {type(env.rng)}")
            
        self.models[key] = model
        self.envs[key] = env
        return model, env

    def solve_instance(self, instance_idx, num_realizations=1):
        # Determine if TWCVRP
        is_tw = self.time_windows is not None and len(self.time_windows) > 0
        
        # Load model and env BEFORE timing to exclude loading time
        model, env = self._get_model_and_env(is_tw)
        if model is None:
            return self._create_empty_result()
            
        start_time = time.time()
            
        # Get data
        depot_indices, customer_indices = self.get_depots_and_customers(instance_idx)
        
        # Retrieve actual coordinates
        if isinstance(self.locations, list):
             instance_locations = self.locations[instance_idx]
        else:
             instance_locations = self.locations[instance_idx]
             
        demands = self._get_demands(instance_idx)
        capacity = self._get_vehicle_capacities(instance_idx)[0]

        # DEBUG: Check data ranges
        print(f"DEBUG: locs range: min={instance_locations.min()}, max={instance_locations.max()}")
        print(f"DEBUG: demands range: min={demands.min()}, max={demands.max()}, cap={capacity}")

        # Only use the first depot for Single Depot VRP formulation
        # Even if the instance has multiple depots (demand=0 nodes), we treat it as Single Depot
        # rooted at the first depot, and ignore other depots (do not visit them as customers).
        main_depot_idx = depot_indices[0]
        
        # depots = instance_locations[depot_indices] # OLD: All depots
        depots = instance_locations[[main_depot_idx]] # NEW: Only first depot
        
        customers = instance_locations[customer_indices]
        dist_matrix = self.dist_matrices[instance_idx]
        
        # Prepare TensorDict
        # Normalize locations to [0, 1]
        # Assuming data is in [0, 1000] or similar large scale
        # We use a fixed scale of 1000 for consistency with standard benchmarks if max > 1
        max_coord = instance_locations.max()
        scale_factor = 1.0
        if max_coord > 1.0:
            scale_factor = 1000.0 # Standard VRP scale
            # Or use max_coord? Standard benchmarks usually are 0-100 or 0-1000.
            # Let's use 1000 as it fits the data range [0, 1000] seen in logs.
        
        # locs: (1, N+1, 2)
        locs = np.concatenate([depots, customers], axis=0)
        
        # DEBUG: Print shapes and indices
        if instance_idx == 0:
            print(f"DEBUG: depot_indices (all): {depot_indices}")
            print(f"DEBUG: main_depot_idx: {main_depot_idx}")
            print(f"DEBUG: customer_indices (first 10): {customer_indices[:10]}")
            print(f"DEBUG: locs shape: {locs.shape}")
            print(f"DEBUG: demands shape: {demands.shape}")
            print(f"DEBUG: capacity: {capacity}")
        
        locs_norm = locs / scale_factor
        
        locs_tensor = torch.tensor(locs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Customers only tensor for RCVRPEnv
        # customers_tensor = torch.tensor(customers, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # demand: (1, N)
        # demands includes depot (size N+1), so we slice it to get customers only (size N)
        # Normalize demands to [0, 1] by dividing by capacity
        # demands_norm = demands[1:] / capacity # OLD: Assumes only index 0 is depot
        demands_norm = demands[customer_indices] / capacity # NEW: Use actual customer indices
        demand_tensor = torch.tensor(demands_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # distance_matrix: (1, N+1, N+1)
        # We must slice the distance matrix to match the selected nodes (main depot + customers)
        all_indices = np.concatenate([[main_depot_idx], customer_indices])
        dist_matrix_reduced = dist_matrix[np.ix_(all_indices, all_indices)]
        
        dist_tensor = torch.tensor(dist_matrix_reduced / scale_factor, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        # Set capacity in generator (hack for RCVRPEnv)
        # If we normalized demands, capacity is effectively 1.0
        env.generator.vehicle_capacity = 1.0
        
        td_data = {
            "locs": locs_tensor, # RMTVRPEnv expects locs to include depot?
            # RCVRPEnv._reset: "locs": torch.cat((td["depot"][:, None, :], td["locs"]), dim=-2)
            # RMTVRPEnv._reset: "locs": td["locs"]
            # So RMTVRPEnv expects locs to include depot.
            # RCVRPEnv expects locs to be customers only.
            
            "depot": locs_tensor[:, 0, :],
            "distance_matrix": dist_tensor
        }
        
        if is_tw:
            # RMTVRPEnv setup
            td_data["locs"] = locs_tensor # Include depot
            td_data["demand_linehaul"] = demand_tensor # Customers only
            
            # Time windows
            tws = self.get_time_windows(instance_idx)
            num_nodes = len(locs)
            tw_array = np.zeros((num_nodes, 2))
            for i in range(num_nodes):
                if i in tws:
                    tw_array[i] = tws[i]
                else:
                    tw_array[i] = [0, 100000]
            
            tw_tensor = torch.tensor(tw_array, dtype=torch.float32, device=self.device).unsqueeze(0)
            td_data["time_windows"] = tw_tensor
            
            # Service times (assume 0)
            td_data["service_time"] = torch.zeros((1, num_nodes), dtype=torch.float32, device=self.device)
            
            # Duration matrix
            # Use distance matrix as duration matrix if time_matrix not available
            # In vrp_base, time_matrix is usually not loaded for standard benchmarks unless specified.
            # We assume duration = distance.
            td_data["duration_matrix"] = dist_tensor
            
        else:
            # RCVRPEnv setup
            td_data["locs"] = locs_tensor[:, 1:, :] # Customers only
            td_data["demand"] = demand_tensor
            
        td_init = TensorDict(td_data, batch_size=[1], device=self.device)
        
        # Reset env
        td = env.reset(td_init)
        
        # Run policy
        with torch.no_grad():
            # decode_type="greedy" or "sampling"?
            # Benchmark usually uses greedy or sampling with N realizations.
            # If num_realizations > 1, we should use sampling?
            # But RRNCO might be trained for greedy.
            # Let's use greedy for now as it is faster and standard for "zero-shot".
            # Or if num_realizations > 1, we can use sampling.
            
            decode_type = "greedy" if num_realizations == 1 else "sampling"
            
            # If sampling, we need to replicate batch?
            # Or policy handles num_starts?
            # RRNet has multistart.
            
            out = model.policy(td, env=env, decode_type=decode_type)
        
        # Parse output
        actions = out["actions"][0].cpu().numpy()
        
        # Map actions (indices in locs) back to original indices
        # locs was constructed as [depots, customers]
        # So index 0..num_depots-1 are depots
        # Index num_depots.. are customers
        
        # num_depots = len(depot_indices) # OLD: All depots
        num_depots = 1 # NEW: Only 1 depot used in locs
        
        def map_node_index(node_idx):
            if node_idx < num_depots:
                # return int(depot_indices[node_idx]) # OLD
                return int(main_depot_idx) # NEW: Always map 0 to main_depot_idx
            else:
                # It's a customer
                # The customer index in 'customers' array is node_idx - num_depots
                # The original index is customer_indices[node_idx - num_depots]
                return int(customer_indices[node_idx - num_depots])
        
        # Convert to routes
        routes = []
        # Start with the first depot (usually there's only one or we pick the first)
        # start_depot = int(depot_indices[0]) # OLD
        start_depot = int(main_depot_idx) # NEW
        current_route = [start_depot] 
        
        for node in actions:
            original_idx = map_node_index(node)
            
            # Check if it's a depot
            is_depot = False
            if node < num_depots:
                is_depot = True
            
            if is_depot:
                if len(current_route) > 1:
                    current_route.append(original_idx) # End with this depot
                    routes.append(current_route)
                    current_route = [original_idx] # Start new route with this depot
            else:
                current_route.append(original_idx)
                
        if len(current_route) > 1:
            # Close the last route with the start depot (or should it be the last visited depot?)
            # Standard VRP usually returns to the same depot.
            current_route.append(start_depot)
            routes.append(current_route)
            
        # Calculate cost
        result = self.calculate_solution_cost(routes, instance_idx, num_realizations)
        end_time = time.time()
        result['runtime'] = end_time - start_time
        return result

