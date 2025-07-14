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
            stepsize : float = 1.0,
            num_workers: int = 1,
            h : float = 1.0,
            num_directions : int = 1,
            beta : float = 0.001,
            patience : int = 1,
            eps : float = 0.1,
            seed: int = 424242,
            popsize = 10,
            reduce_size = False,
            no_exploit = False,
            max_num_sections = 100,
            max_content_size = int(2**15),
            max_ls_steps = 10,
            armijo_constant : float = 1e-5,
            min_stepsize : float = 1e-5,
            max_stepsize : float = 1e3,
            contraction_factor : float = 0.5,
            expansion_factor : float = 2.0,
            only_printable_char : bool = False,
            method: str = "zexe", # can be random-search, zexe
            expl_strat = 'rs',
            exploration_strategy: ExplorationStrategyID = ExplorationStrategyID.RANDOM,
            #which_sections: list = None,
            y_target: Union[int, None] = None,
            random_init: bool = False,
            trackers: Union[List[Tracker], Tracker] = None,
    ):
      #  if which_sections is None:
      #      which_sections = ['rodata']
        initializer = IdentityInitializer(
            random_init=random_init
        )

        self.content_size = perturbation_size // how_many_sections
        # TODO: change this to ZEXESectionIjectionManipulation
        manipulation_function = ZEXESectionInjectionManipulation(
            how_many_sections=how_many_sections, 
            only_printable_char = only_printable_char,
            content_size=self.content_size)

        if method == "random_search":
            optimizer_cls = MalwareOptimizerFactory.create_rs(
                seed=seed
            )
        elif method == "zexe":
            optimizer_cls = MalwareOptimizerFactory.create_zexe(
                stepsize=stepsize,
                h=h,
                eps=eps,
                patience=patience,
                exploration_strategy=exploration_strategy,
                max_stepsize = max_stepsize,
                armijo_constant = armijo_constant,
                num_directions = num_directions,
                min_stepsize = min_stepsize,
                reduce_size = reduce_size,
                popsize = popsize,
                contraction_factor = contraction_factor,
                expansion_factor = expansion_factor,
                manipulation_function=manipulation_function,
                beta=beta,
                seed=seed
            )
        elif method == "zexers":
            optimizer_cls = MalwareOptimizerFactory.create_zexers(
                stepsize=stepsize,
                h=h,
                eps=eps,
                patience=patience,
                max_stepsize = max_stepsize,
                armijo_constant = armijo_constant,
                num_directions = num_directions,
                min_stepsize = min_stepsize,
                reduce_size = reduce_size,
                popsize = popsize, expl_strategy=expl_strat,
                contraction_factor = contraction_factor,
                expansion_factor = expansion_factor,
                manipulation_function=manipulation_function,
                beta=beta,
                seed=seed
            )
        elif method == "dzexe":
            optimizer_cls = MalwareOptimizerFactory.create_dzexe(
                num_directions=num_directions,
                manipulation_function=manipulation_function,
                armijo_constant = armijo_constant,
                max_num_sections = max_num_sections,
                max_content_size = max_content_size,
                max_ls_steps = max_ls_steps,
                no_exploit = no_exploit,
                popsize=popsize,
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
