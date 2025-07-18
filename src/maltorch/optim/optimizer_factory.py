from functools import partial
from typing import Union, Callable

from nevergrad.optimization.differentialevolution import (
    DifferentialEvolution,
)

from maltorch.optim.bgd import BGD
from maltorch.optim.random_search import RSOptimizer
from maltorch.optim.zexe import ZEXEOptimizer, ExplorationStrategyID
from maltorch.optim.zexe_rs import ZEXERSOptimizer
from maltorch.optim.byte_gradient_processing import ByteGradientProcessing
from maltorch.optim.dzexe import DZEXEOptimizer

class MalwareOptimizerFactory:
    @staticmethod
    def create(optim_cls: Union[str, Callable], **optimizer_args):
        if not isinstance(optim_cls, str):
            return partial(optim_cls, **optimizer_args)()
        if optim_cls == "bgd":
            return MalwareOptimizerFactory.create_bgd(**optimizer_args)
        if optim_cls == "ga":
            return MalwareOptimizerFactory.create_ga(**optimizer_args)
        if optim_cls == "zexe":
            return MalwareOptimizerFactory.create_zexe(**optimizer_args)
        if optim_cls == "rs":
            return MalwareOptimizerFactory.create_rs(**optimizer_args)
        raise NotImplementedError(f"Optimizer {optim_cls} not included.")

    @staticmethod
    def create_bgd(lr: int, device: str = "cpu") -> partial[BGD]:
        return partial(
            BGD,
            lr=lr,
            gradient_processing=ByteGradientProcessing(),
            device=device,
        )

    @staticmethod
    def create_rs(seed: int = 42) -> RSOptimizer:
        return RSOptimizer(seed=seed)

    @staticmethod
    def create_ga(population_size: int = 10) -> DifferentialEvolution:
        return DifferentialEvolution(popsize=population_size, crossover="twopoints")

    @staticmethod
    def create_zexe(
                    stepsize: float = 1.0, 
                    h : float = 1.0,
                    num_directions : int = 1,
                    armijo_constant : float = 1e-5,
                    min_stepsize : float = 1e-5,
                    max_stepsize : float = 1e3,
                    contraction_factor : float = 0.5,
                    expansion_factor : float = 2.0,
                    exploration_strategy: ExplorationStrategyID = ExplorationStrategyID.RANDOM,
                    expl_strategy_params: dict = {},     
                    seed: int = 42) -> ZEXEOptimizer:
        return ZEXEOptimizer(
                        stepsize = stepsize, 
                        h = h,
                        num_directions= num_directions,
                        armijo_constant = armijo_constant,
                        min_stepsize = min_stepsize,
                        max_stepsize  = max_stepsize,
                        contraction_factor = contraction_factor,
                        expansion_factor  = expansion_factor,
                        exploration_strategy = exploration_strategy,
                        expl_strategy_params = expl_strategy_params,     
                        seed=seed)

