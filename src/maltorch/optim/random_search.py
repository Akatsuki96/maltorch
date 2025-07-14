from typing import Callable, Optional
import numpy as np
import torch.optim
from torch import Tensor
from enum import Enum

import nevergrad as ng
from nevergrad.optimization.base import Optimizer, ConfiguredOptimizer
import nevergrad.common.typing as tp
from nevergrad.parametrization import parameter as p
from nevergrad.optimization.differentialevolution import (
    DifferentialEvolution, _DE
)





class _RS(Optimizer): # It has to extend optimizer (ConfiguredOptimizer or Optimizer?) class of nevergrad?


    def __init__(self, parametrization, 
                    budget: tp.Optional[int] = None, 
                    num_workers: int = 1,
                    *,
                    config):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

        self._config = ZEXEOptimizer() if config is None else config
        self.best = None


    def _internal_ask_candidate(self) -> p.Parameter:
        return self.parametrization.sample().set_integer_casting()

    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:

        if self.best is None or loss < self.best[1]:
            self.best = (candidate, loss)

    def recommend(self) -> p.Parameter:
        return self.best[0]






class RSOptimizer(ConfiguredOptimizer):

    def __init__(self, seed: int = 42):
        self.name = "RS"
        super().__init__(_RS, locals(), as_config=True)
        self.seed=seed


