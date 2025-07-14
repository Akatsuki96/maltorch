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
        if optim_cls == "zexers":
            return MalwareOptimizerFactory.create_zexers(**optimizer_args)
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
    def create_zexe(stepsize: float = 1.0, 
                    h : float = 1.0,
                    num_directions : int = 1,
                    beta : float = 0.001,
                    eps : float = 0.1,
                    patience : int = 1,
                    armijo_constant : float = 1e-5,
                    min_stepsize : float = 1e-5,
                    max_stepsize : float = 1e3,
                    contraction_factor : float = 0.5,
                    expansion_factor : float = 2.0,
                    reduce_size : bool = False,
                    popsize = 10,
                    manipulation_function: Callable = None,
                    exploration_strategy : ExplorationStrategyID = ExplorationStrategyID.RANDOM,
                    seed: int = 42) -> ZEXEOptimizer:
        expl_params = {'eps' : eps, 'patience' : patience, 'beta': beta, 'seed' : seed}
        print("[--] ZEXE OPT: ", stepsize)
        return ZEXEOptimizer(stepsize=stepsize, 
                            h=h,
                            max_stepsize = max_stepsize,
                            armijo_constant = armijo_constant,
                            min_stepsize = min_stepsize, popsize=popsize,
                            contraction_factor = contraction_factor,
                            expansion_factor = expansion_factor, manipulation_function=manipulation_function,
                            num_directions=num_directions, reduce_size=reduce_size,
                            beta=beta, expl_strategy_params=expl_params,
                            exploration_strategy = exploration_strategy, 
                            seed=seed)

    @staticmethod
    def create_zexers(stepsize: float = 1.0, 
                    h : float = 1.0,
                    num_directions : int = 1,
                    beta = lambda k : 0.001 * np.log(k + 1) / (k + 1),
                    eps : float = 0.1,
                    patience : int = 1,
                    armijo_constant : float = 1e-5,
                    min_stepsize : float = 1e-5,
                    max_stepsize : float = 1e3,
                    contraction_factor : float = 0.5,
                    expansion_factor : float = 2.0,
                    expl_strategy  = 'rs',
                    reduce_size : bool = False,
                    popsize = 10,
                    manipulation_function: Callable = None,
                    seed: int = 42) -> ZEXERSOptimizer:
        expl_params = {'eps' : eps, 'patience' : patience, 'beta': beta, 'seed' : seed}
        print("[--] ZEXERS OPT: ", stepsize)
        return ZEXERSOptimizer(stepsize=stepsize, 
                            h=h,
                            max_stepsize = max_stepsize,
                            armijo_constant = armijo_constant,
                            min_stepsize = min_stepsize, popsize=popsize,
                            contraction_factor = contraction_factor, expl_strategy = expl_strategy,
                            expansion_factor = expansion_factor, manipulation_function=manipulation_function,
                            num_directions=num_directions, reduce_size=reduce_size,
                            beta=beta, 
                            seed=seed)
    @staticmethod
    def create_dzexe(num_directions : int = 1,
                    manipulation_function = None,
                    armijo_constant : float = 1e-5,
                    max_num_sections : int = 100,
                    max_content_size : int = 65536,
                    popsize : int = 10,
                    no_exploit = False,
                    max_ls_steps = 10,
                    seed: int = 42):
        return DZEXEOptimizer(num_directions=num_directions,
                            manipulation_function=manipulation_function,
                            armijo_constant = armijo_constant,
                            max_num_sections = max_num_sections,
                            max_content_size = max_content_size,
                            popsize=popsize, no_exploit = no_exploit,
                            max_ls_steps = max_ls_steps,
                            seed=seed)