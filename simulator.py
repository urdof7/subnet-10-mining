import numpy as np
from typing import Dict, Union
from decimal import Decimal
import typing


# from sturdy.utils.misc import borrow_rate, check_allocations
# from sturdy.pools import (
#     generate_assets_and_pools,
#     generate_initial_allocations_for_pools,
# )
# from sturdy.constants import *
import copy

NUM_POOLS = 10  # number of pools to generate per query per epoch - for scoring miners
MIN_BASE_RATE = 0.01
MAX_BASE_RATE = 0.05  # keep the base rate the same for every pool for now - 0
BASE_RATE_STEP = 0.01
MIN_SLOPE = 0.01
MAX_SLOPE = 0.1
MIN_KINK_SLOPE = 0.15
MAX_KINK_SLOPE = 1
SLOPE_STEP = 0.001
MIN_OPTIMAL_RATE = 0.65
MAX_OPTIMAL_RATE = 0.95
OPTIMAL_UTIL_STEP = 0.05
MIN_UTIL_RATE = 0.55
MAX_UTIL_RATE = 0.95
UTIL_RATE_STEP = 0.05
TOTAL_ASSETS = 2.0  # total assets to allocate ( set to 1 for simplicity :^) )
CHUNK_RATIO = 0.01  # chunk size as a percentage of total assets allocated during each iteration of greedy allocation algorithm
GREEDY_SIG_FIGS = 8  # significant figures to round to for greedy algorithm allocations

REVERSION_SPEED = 0.1  # reversion speed to median borrow rate of pools
TIMESTEPS = 50  # simulation timesteps
STOCHASTICITY = 0.025  # stochasticity - some randomness to sprinkle into the simulation
POOL_RESERVE_SIZE = 1.0

QUERY_RATE = 2  # how often synthetic validator queries miners (blocks)
QUERY_TIMEOUT = 10  # timeout (seconds)

# latency reward curve scaling parameters
STEEPNESS = 1.0
DIV_FACTOR = 1.5  # a scaling factor

def randrange_float(
    start,
    stop,
    step,
    sig: int = GREEDY_SIG_FIGS,
    max_prec: int = GREEDY_SIG_FIGS,
    rng_gen=np.random,
):
    num_steps = int((stop - start) / step)
    random_step = rng_gen.randint(0, num_steps + 1)
    return format_num_prec(start + random_step * step, sig=sig, max_prec=max_prec)

def format_num_prec(
    num: float, sig: int = GREEDY_SIG_FIGS, max_prec: int = GREEDY_SIG_FIGS
) -> float:
    return float(f"{{0:.{max_prec}f}}".format(float(format(num, f".{sig}f"))))

def borrow_rate(util_rate: float, pool: Dict) -> float:
    interest_rate = (
        pool["base_rate"] + (util_rate / pool["optimal_util_rate"]) * pool["base_slope"]
        if util_rate < pool["optimal_util_rate"]
        else pool["base_rate"]
        + pool["base_slope"]
        + ((util_rate - pool["optimal_util_rate"]) / (1 - pool["optimal_util_rate"]))
        * pool["kink_slope"]
    )

    return interest_rate


def supply_rate(util_rate: float, pool: Dict) -> float:
    return util_rate * borrow_rate(util_rate, pool)


def check_allocations(
    assets_and_pools: Dict[str, Union[Dict[str, float], float]],
    allocations: Dict[str, float],
) -> bool:
    """
    Checks allocations from miner.

    Args:
    - assets_and_pools (Dict[str, Union[Dict[str, float], float]]): The assets and pools which the allocations are for.
    - allocations (Dict[str, float]): The allocations to validate.

    Returns:
    - bool: Represents if allocations are valid.
    """

    # Ensure the allocations are provided and valid
    if not allocations or not isinstance(allocations, Dict):
        return False

    # Ensure the 'total_assets' key exists in assets_and_pools and is a valid number
    to_allocate = assets_and_pools.get("total_assets")
    if to_allocate is None or not isinstance(to_allocate, (int, float)):
        return False

    to_allocate = Decimal(str(to_allocate))
    total_allocated = Decimal(0)

    # Check allocations
    for _, allocation in allocations.items():
        try:
            allocation_value = Decimal(str(allocation))
        except (ValueError, TypeError):
            return False

        if allocation_value < 0:
            return False

        total_allocated += allocation_value

        if total_allocated > to_allocate:
            return False

    # Ensure total allocated does not exceed the total assets
    if total_allocated > to_allocate:
        return False

    return True

def generate_assets_and_pools(rng_gen=np.random) -> typing.Dict:  # generate pools
    assets_and_pools = {}
    pools = {
        str(x): {
            "pool_id": str(x),
            "base_rate": randrange_float(
                MIN_BASE_RATE, MAX_BASE_RATE, BASE_RATE_STEP, rng_gen=rng_gen
            ),
            "base_slope": randrange_float(
                MIN_SLOPE, MAX_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),
            "kink_slope": randrange_float(
                MIN_KINK_SLOPE, MAX_KINK_SLOPE, SLOPE_STEP, rng_gen=rng_gen
            ),  # kink rate - kicks in after pool hits optimal util rate
            "optimal_util_rate": randrange_float(
                MIN_OPTIMAL_RATE, MAX_OPTIMAL_RATE, OPTIMAL_UTIL_STEP, rng_gen=rng_gen
            ),  # optimal util rate - after which the kink slope kicks in
            "borrow_amount": format_num_prec(
                POOL_RESERVE_SIZE
                * randrange_float(
                    MIN_UTIL_RATE, MAX_UTIL_RATE, UTIL_RATE_STEP, rng_gen=rng_gen
                )
            ),  # initial borrowed amount from pool
            "reserve_size": POOL_RESERVE_SIZE,
        }
        for x in range(NUM_POOLS)
    }

    assets_and_pools["total_assets"] = TOTAL_ASSETS
    assets_and_pools["pools"] = pools

    return assets_and_pools


# generate intial allocations for pools
def generate_initial_allocations_for_pools(
    assets_and_pools: typing.Dict, size: int = NUM_POOLS, rng_gen=np.random
) -> typing.Dict:
    nums = np.ones(size)
    allocs = nums / np.sum(nums) * assets_and_pools["total_assets"]
    allocations = {str(i): alloc for i, alloc in enumerate(allocs)}

    return allocations



class Simulator(object):
    def __init__(
        self,
        # config,
        timesteps=TIMESTEPS,
        reversion_speed=REVERSION_SPEED,
        stochasticity=STOCHASTICITY,
        seed=None,
    ):
        self.timesteps = timesteps
        self.reversion_speed = reversion_speed
        self.stochasticity = stochasticity
        self.assets_and_pools = {}
        self.allocations = {}
        self.pool_history = []
        self.init_rng = None
        self.rng_state_container = None
        self.seed = seed

    # initializes data - by default these are randomly generated
    def init_data(
        self,
        init_assets_and_pools: Dict[str, Union[Dict[str, float], float]] = None,
        init_allocations: Dict[str, float] = None,
    ):
        if self.rng_state_container is None or self.init_rng is None:
            raise RuntimeError(
                "You must have first initialize()-ed the simulation if you'd like to initialize some data"
            )

        if init_assets_and_pools is None:
            self.assets_and_pools = generate_assets_and_pools(
                rng_gen=self.rng_state_container
            )
        else:
            self.assets_and_pools = init_assets_and_pools

        if init_allocations is None:
            self.allocations = generate_initial_allocations_for_pools(
                self.assets_and_pools, rng_gen=self.rng_state_container
            )
        else:
            self.allocations = init_allocations

        # initialize pool history
        self.pool_history = [
            {
                uid: {
                    "borrow_amount": pool["borrow_amount"],
                    "reserve_size": pool["reserve_size"],
                    "borrow_rate": borrow_rate(
                        pool["borrow_amount"] / pool["reserve_size"], pool
                    ),
                }
                for uid, pool in self.assets_and_pools["pools"].items()
            }
        ]

    # initialize fresh simulation instance
    def initialize(self):
        # create fresh rng state
        self.init_rng = np.random.RandomState(self.seed)
        self.rng_state_container = copy.copy(self.init_rng)

    # reset sim to initial params for rng
    def reset(self):
        if self.rng_state_container is None or self.init_rng is None:
            raise RuntimeError(
                "You must have first initialize()-ed the simulation if you'd like to reset it"
            )
        self.rng_state_container = copy.copy(self.init_rng)

    # update the reserves in the pool with given allocations
    def update_reserves_with_allocs(self, allocs=None):
        if (
            len(self.pool_history) != 1
            or len(self.assets_and_pools) <= 0
            or len(self.allocations) <= 0
        ):
            raise RuntimeError(
                "You must first initialize() and init_data() before running the simulation!!!"
            )

        if allocs is None:
            allocations = self.allocations
        else:
            allocations = allocs

        check_allocations(self.assets_and_pools, allocations)

        if len(self.pool_history) != 1:
            raise RuntimeError(
                "You must have first init data for the simulation if you'd like to update reserves"
            )

        for uid, alloc in allocations.items():
            pool = self.assets_and_pools["pools"][uid]
            pool_history_start = self.pool_history[0]
            pool["reserve_size"] += alloc
            pool_from_history = pool_history_start[uid]
            pool_from_history["reserve_size"] += allocations[uid]
            pool_from_history["borrow_rate"] = borrow_rate(
                pool["borrow_amount"] / pool["reserve_size"], pool
            )

    # initialize pools
    # Function to update borrow amounts and other pool params based on reversion rate and stochasticity
    def generate_new_pool_data(self):
        latest_pool_data = self.pool_history[-1]
        curr_borrow_rates = np.array(
            [pool["borrow_rate"] for _, pool in latest_pool_data.items()]
        )
        curr_borrow_amounts = np.array(
            [pool["borrow_amount"] for _, pool in latest_pool_data.items()]
        )
        curr_reserve_sizes = np.array(
            [pool["reserve_size"] for _, pool in latest_pool_data.items()]
        )

        median_rate = np.median(curr_borrow_rates)  # Calculate the median borrow rate
        noise = self.rng_state_container.normal(
            0, self.stochasticity, len(curr_borrow_rates)
        )  # Add some random noise
        rate_changes = (
            -self.reversion_speed * (curr_borrow_rates - median_rate) + noise
        )  # Mean reversion principle
        new_borrow_amounts = (
            curr_borrow_amounts + rate_changes * curr_borrow_amounts
        )  # Update the borrow amounts
        amounts = np.clip(
            new_borrow_amounts, 0, curr_reserve_sizes
        )  # Ensure borrow amounts do not exceed reserves
        pool_uids = list(latest_pool_data.keys())

        new_pool_data = {
            pool_uids[i]: {
                "borrow_amount": amounts[i],
                "reserve_size": curr_reserve_sizes[i],
                "borrow_rate": borrow_rate(
                    amounts[i] / curr_reserve_sizes[i],
                    self.assets_and_pools["pools"][pool_uids[i]],
                ),
            }
            for i in range(len(amounts))
        }

        return new_pool_data

    # run simulation
    def run(self):
        if len(self.pool_history) != 1:
            raise RuntimeError(
                "You must first initialize() and init_data() before running the simulation!!!"
            )
        for _ in range(1, self.timesteps):
            new_info = self.generate_new_pool_data()
            # TODO: do we need to copy?
            self.pool_history.append(new_info.copy())


simulator = Simulator()
simulator.initialize()
simulator.init_data()
simulator.run()

question = simulator.generate_new_pool_data()
print(question)

