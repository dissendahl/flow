"""Multi-agent traffic light example (single shared policy)."""

from ray.rllib.agents.ddpg.ddpg_policy import DDPGTFPolicy
from flow.envs.multiagent import MultiTrafficLightGridPOEnv
from flow.networks import TrafficLightGridNetwork
from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams
from flow.core.params import InFlows, SumoCarFollowingParams, VehicleParams
from flow.controllers import SimCarFollowingController, GridRouter
from ray.tune.registry import register_env
from flow.utils.registry import make_create_env

# Experiment parameters
N_ROLLOUTS = 20  # number of rollouts per training iteration
N_CPUS = 8  # number of parallel workers

# Environment parameters
HORIZON = 250  # time horizon of a single rollout
V_ENTER = 20  # enter speed for departing vehicles
V_TARGET = 40 # target velocity /speed
INNER_LENGTH = 300  # length of inner edges in the traffic light grid network
LONG_LENGTH = 100  # length of final edge in route
SHORT_LENGTH = 300  # length of edges that vehicles start on
# number of vehicles originating in the left, right, top, and bottom edges
N_LEFT, N_RIGHT, N_TOP, N_BOTTOM = 10, 10, 10, 10

USE_INFLOW = True
EDGE_INFLOW = 300  # inflow rate of vehicles at every edge
N_ROWS = 3  # number of row of bidirectional lanes
N_COLUMNS = 3  # number of columns of bidirectional lanes
LANE_NUMBER = 1 # number of lanes

SWITCH_TIME = 3
CARS_OBSERVED = 2
DISCRETE = False
LOCAL_EDGES =  4
LOCAL_LIGHTS = 4

SIM_STEP = 1 # Simulation step size


# we place a sufficient number of vehicles to ensure they confirm with the
# total number specified above. We also use a "right_of_way" speed mode to
# support traffic light compliance
vehicles = VehicleParams()
num_vehicles = (N_LEFT + N_RIGHT) * N_COLUMNS + (N_BOTTOM + N_TOP) * N_ROWS
vehicles.add(
    veh_id="human",
    acceleration_controller=(SimCarFollowingController, {}),
    routing_controller=(GridRouter, {}),
    car_following_params=SumoCarFollowingParams(
        min_gap=2.5,
        max_speed=V_ENTER,
        decel=7.5,  # avoid collisions at emergency stops
        speed_mode="right_of_way",
    ),
    num_vehicles=num_vehicles)

# inflows of vehicles are place on all outer edges (listed here)
outer_edges = []
outer_edges += ["left{}_{}".format(N_ROWS, i) for i in range(N_COLUMNS)]
outer_edges += ["right0_{}".format(i) for i in range(N_ROWS)]
outer_edges += ["bot{}_0".format(i) for i in range(N_ROWS)]
outer_edges += ["top{}_{}".format(i, N_COLUMNS) for i in range(N_ROWS)]

# equal inflows for each edge (as dictate by the EDGE_INFLOW constant)
inflow = InFlows()
for edge in outer_edges:
    inflow.add(
        veh_type="human",
        edge=edge,
        vehs_per_hour=EDGE_INFLOW,
        depart_lane="free",
        depart_speed=V_ENTER)

flow_params = dict(
    # name of the experiment (for multiple policies!)
    exp_tag="multimultiagent-traffic-light",

    # name of the flow environment the experiment is running on
    env_name=MultiTrafficLightGridPOEnv,

    # name of the network class the experiment is running on
    network=TrafficLightGridNetwork,

    # simulator that is used by the experiment
    simulator='traci',

    # sumo-related parameters (see flow.core.params.SumoParams)
    sim=SumoParams(
        restart_instance=True,
        sim_step=SIM_STEP,
        render=False,
    ),

    # environment related parameters (see flow.core.params.EnvParams)
    env=EnvParams(
        horizon=HORIZON,
        additional_params={
            "target_velocity": V_TARGET,
            "switch_time": SWITCH_TIME,
            "num_observed": CARS_OBSERVED,
            "discrete": DISCRETE,
            "tl_type": "actuated",
            "num_local_edges": LOCAL_EDGES,
            "num_local_lights": LOCAL_LIGHTS,
        },
    ),

    # network-related parameters (see flow.core.params.NetParams and the
    # network's documentation or ADDITIONAL_NET_PARAMS component)
    net=NetParams(
        inflows=inflow,
        additional_params={
            "speed_limit": V_TARGET,  # inherited from grid0 benchmark
            "grid_array": {
                "short_length": SHORT_LENGTH,
                "inner_length": INNER_LENGTH,
                "long_length": LONG_LENGTH,
                "row_num": N_ROWS,
                "col_num": N_COLUMNS,
                "cars_left": N_LEFT,
                "cars_right": N_RIGHT,
                "cars_top": N_TOP,
                "cars_bot": N_BOTTOM,
            },
            "horizontal_lanes": LANE_NUMBER,
            "vertical_lanes": LANE_NUMBER,
        },
    ),

    # vehicles to be placed in the network at the start of a rollout (see
    # flow.core.params.VehicleParams)
    veh=vehicles,

    # parameters specifying the positioning of vehicles upon initialization
    # or reset (see flow.core.params.InitialConfig)
    initial=InitialConfig(
        spacing='custom',
        shuffle=True,
    ),
)

create_env, env_name = make_create_env(params=flow_params, version=0)

# Register as rllib env
register_env(env_name, create_env)

test_env = create_env()
obs_space = test_env.observation_space
act_space = test_env.action_space


def gen_policy():
    """Generate a policy in RLlib."""
    return DDPGTFPolicy, obs_space, act_space, {}


# Setup PG with a single policy graph for all agents
POLICY_GRAPHS = {'center0': gen_policy(),
                'center1': gen_policy(),
                'center2': gen_policy(),
                'center3': gen_policy(),
                'center4': gen_policy(),
                'center5': gen_policy(),
                'center6': gen_policy(),
                'center7': gen_policy(),
                'center8': gen_policy(),}

# policy key string
def policy_mapping_fn(s):
    return s


POLICIES_TO_TRAIN = ['center0', 'center1', 'center2', 'center3', 'center4', 'center5', 'center6', 'center7', 'center8']
