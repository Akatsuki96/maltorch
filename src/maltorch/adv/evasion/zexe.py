from pathlib import Path
from typing import Type, Union, List, Callable

from secmlt.trackers import Tracker
from torch.nn import BCEWithLogitsLoss

from maltorch.adv.evasion.base_optim_attack_creator import (
    BaseOptimAttackCreator,
    OptimizerBackends,
)
from maltorch.adv.evasion.gradfree_attack import GradientFreeBackendAttack
from maltorch.initializers.initializers import IdentityInitializer
from maltorch.manipulations.zexe_section_injection_manipulation import ZEXESectionInjectionManipulation
from maltorch.optim.optimizer_factory import MalwareOptimizerFactory
from maltorch.optim.zexe import ExplorationStrategyID


class ZEXESectionInjection(GradientFreeBackendAttack):

    def __init__(
            self,
            query_budget: int,
            perturbation_size: int,
            how_many_sections: int,
            num_workers: int = 1,
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
            only_printable_char = False,
            seed = 123141,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
      #  if which_sections is None:
      #      which_sections = ['rodata']
        initializer = IdentityInitializer(
            random_init=random_init
        )

        self.content_size = perturbation_size // how_many_sections - 8
        # TODO: change this to ZEXESectionIjectionManipulation
        manipulation_function = ZEXESectionInjectionManipulation(
            how_many_sections=how_many_sections, 
            only_printable_char = only_printable_char,
            content_size=self.content_size)

        optimizer_cls = MalwareOptimizerFactory.create_zexe(
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
            seed=seed
        )
        loss_function = BCEWithLogitsLoss(reduction="none")

        
        # GAMMASectionInjectionManipulation(benignware_folder=benignware_folder,
        #                                                           which_sections=which_sections,
        #                                                           how_many_sections=how_many_sections)
        super().__init__(
            y_target=y_target,
            query_budget=query_budget,
            loss_function=loss_function,
            initializer=initializer,
            manipulation_function=manipulation_function,
            optimizer_cls=optimizer_cls,
            trackers=trackers,
        )
