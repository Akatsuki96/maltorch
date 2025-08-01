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


class ZEXEPhase(Enum):
    ITERATE = 0
    FORWARD_DIRECTION = 1
    LINE_SEARCH_STEP = 2
    EXPLORATION = 3

class ExplorationStrategyID(Enum):
    NOEXPLORATION = 0
    RANDOM = 1
    LANGEVIN = 2
    ANNEALING = 3
    GA = 4


class ExplorationStrategy:

    def mutate(self, x, g, gamma, k, X, y):
        raise NotImplementedError()

class NoExploration(ExplorationStrategy):
    def mutate(self, x, g, gamma, k, X, y):
        return x - gamma * g
    
    def update(self, x, y):
        pass

class LangevinExploration(ExplorationStrategy):
    def __init__(self, beta: float | Callable[[int], float] = 0.1, seed : int = 42):
        self.beta = beta if isinstance(beta, Callable) else lambda _: beta
        self.rnd_state = np.random.RandomState(seed)

    def mutate(self, x, g, gamma, k, X, y):
        z = np.sqrt(2 * self.beta(k) ) * self.rnd_state.randn(x.shape[0], x.shape[1])
        return x - gamma * g + z, False, z

    def update(self, x, y):
        pass


class RandomExploration(ExplorationStrategy):
    def __init__(self, eps : float = 0.1, seed : int = 42):
        self.eps = eps
        self.rnd_state = np.random.RandomState(seed)

    def mutate(self, x, g, gamma, k, X, y):
        if np.linalg.norm(g, ord=2) < self.eps:
            return self.rnd_state.rand(x.shape[0], x.shape[1]) - 0.5, True, 0.0
        return x - gamma * g, False, 0.0
    
    def update(self, x, y):
        pass

class GAExploration(ExplorationStrategy):
    def __init__(self, perturbation_size, popsize : int = 10, eps : float = 1e-5, seed : int = 42):
        self.perturbation_size = perturbation_size
        self.rnd_state = np.random.RandomState(seed)
        self.popsize = popsize
        self.eps = eps
        self.current_elem = None
        self.de = _DE(p.Array(shape=(1, perturbation_size)), config=DifferentialEvolution(popsize=popsize, crossover="twopoints"))

    def mutate(self, x, g, gamma, k, X, y):
        if np.linalg.norm(g, ord=2) < self.eps:
            self.current_elem = self.de.ask()
            return self.current_elem.value, True, 0.0
        return x - gamma * g, False, 0.0
    
    def update(self, x, y):
        if self.current_elem is None:
            self.de.suggest(x)
            self.current_elem = self.de.ask()

        self.de.tell(self.current_elem, y)
        self.current_elem = None


def get_expl_strategy(strategy: ExplorationStrategyID, params: dict):
    if strategy == ExplorationStrategyID.NOEXPLORATION:
        return NoExploration()
    if strategy == ExplorationStrategyID.RANDOM:
        return RandomExploration(eps=params['eps'],  seed=params['seed'])
    if strategy == ExplorationStrategyID.LANGEVIN:
        return LangevinExploration(beta=params['beta'], seed=params['seed'])
    if strategy == ExplorationStrategyID.GA:
        return GAExploration(perturbation_size=params['perturbation_size'], popsize=params['popsize'], seed=params['seed'])




class _ZEXE(Optimizer): # It has to extend optimizer (ConfiguredOptimizer or Optimizer?) class of nevergrad?


    def __init__(self, parametrization, 
                    budget: tp.Optional[int] = None, 
                    num_workers: int = 1,
                    *,
                    config):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

        self._config = ZEXEOptimizer() if config is None else config
        self.num_directions = self._config.num_directions
        self.sec_idx = 0
        self.stepsize = self._config.stepsize
        self.init_stepsize = self._config.stepsize
        self.h = self._config.h
        self.h_0 = self._config.h
        self.exploration_strategy = self._config.exploration_strategy
        self.armijo_constant = self._config.armijo_constant
        self.min_stepsize = self._config.min_stepsize
        self.max_stepsize = self._config.max_stepsize
        self.contraction_factor = self._config.contraction_factor
        self.expansion_factor = self._config.expansion_factor
        self.current_iterate = None
        self.phase = ZEXEPhase.ITERATE
        self.z = 0.0

        self.rnd_state = np.random.RandomState(self._config.seed)
        self.current_idx = 0
        self.d = self.parametrization.value.shape[1]
        self.best = None
        self.k = 0
        self.g = None
        self.X, self.Y = np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)# [], []

    def generate_direction_matrix(self):
        v = self.rnd_state.randn(self.d)
        v /= np.linalg.norm(v, ord=2)
        indices = self.rnd_state.choice(self.d, self.num_directions, replace=False)
        I = np.zeros((self.d, self.num_directions))
        I[indices, range(self.num_directions)] = 1.0
        H = I - 2 * np.outer(v, v[indices])
        return H


    def _grad_ask(self):
        g = np.zeros(self.d)
        for i in range(self.num_directions):
            g += ((self.fvalues[i + 1] - self.fvalues[0]) / self.h ) * self.P[:, i]
        return g


    def _internal_ask_candidate(self) -> p.Parameter:
        if self.phase == ZEXEPhase.FORWARD_DIRECTION:
            new_iterate = self.current_iterate + self.h * self.P[:, self.current_idx].reshape((1, self.d))
            
        if self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            new_iterate = self.current_iterate  - self.stepsize * self.g + self.z


        if self.phase == ZEXEPhase.ITERATE:
            self.g = self._grad_ask()

            self.k += 1
            print(f"[{str(self.phase)}] VALUES: ", self.fvalues)
            self.h = max(self.h / self.k, 0.01)
            if self.h == 0.01:
                self.h = self.h_0
            new_iterate, exploration, self.z = self.exploration_strategy.mutate(self.current_iterate, self.g, self.stepsize, self.k, self.X, self.Y)
            if exploration:
                self.phase = ZEXEPhase.EXPLORATION
            elif np.linalg.norm(self.g) > 1e-3:
                self.phase = ZEXEPhase.LINE_SEARCH_STEP
            
        new_iterate = (255.0*((np.tanh(new_iterate)/2) + 0.5)).astype(np.int64)
        self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)

        return self.parametrization#.spawn_child(new_iterate.astype(np.int64)) 
            
#             if np.linalg.norm(self.g) < 0.01: 
#                 print("EXPLORATION")
#                 self.phase = ZEXEPhase.EXPLORATION
# #                self.num_iters = 0
#                 return self._get_exploration_value()# self.parametrization#new_parametrization#self.parametrization.spawn_child(new_iterate.reshape(1, -1).astype(np.int64)) 
#             else:
#                 print("EXPLOITATION")
#                 self.phase = ZEXEPhase.LINE_SEARCH_STEP
#                 candidate = self.best[0]
#                 x_tilde = (candidate.value / 255.0) - 0.5 # put in [-0.5, 0.5]
#                 x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space

#                 self.current_iterate = x_tilde
#                 self.current_value = self.best[1]
#                 self.manipulation_function.how_many_sections = self.best[2]
#                 self.manipulation_function.content_size = self.best[3]
#                 new_iterate = self.current_iterate - self.stepsize * self.g
#         new_iterate = (255.0*((np.tanh(new_iterate)/2) + 0.5)).astype(np.int64)
#         self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)

#         return self.parametrization#.spawn_child(new_iterate.astype(np.int64)) 


    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        x_tilde = (candidate.value / 255.0) - 0.5 # put in [-0.5, 0.5]
        x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space
        self.exploration_strategy.update(x_tilde, loss)

        if self.best is None or self.best[1] > loss:
            self.best = (candidate, loss)


        if self.g is not None:
            print(f"[{str(self.phase)}] k: {self.k}\tf(x_k): {loss}\tx_k: {candidate.value}\th: {self.h}\t||g||: {np.linalg.norm(self.g)}\tbest: {self.best[1]}")

        if self.current_iterate is None:
            self.current_iterate = x_tilde #candidate.value
            self.current_value = loss
        if self.phase == ZEXEPhase.ITERATE:
            self.phase = ZEXEPhase.FORWARD_DIRECTION
            self.iterates = [x_tilde]
            self.fvalues = [loss]
            self.current_idx = 0
            self.P = self.generate_direction_matrix()
        elif self.phase == ZEXEPhase.FORWARD_DIRECTION:
            self.iterates.append(x_tilde)
            self.fvalues.append(loss )
            self.current_idx += 1
            if self.current_idx == self.num_directions:
                self.phase = ZEXEPhase.ITERATE
                self.current_idx = 0
        elif self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            cond = self.current_value - self.armijo_constant * self.stepsize * np.square(np.linalg.norm(self.g))
            print(f"[{str(self.phase)}] loss: {loss}, cond: {cond}, stepsize: {self.stepsize}")
            if loss  < cond or self.stepsize <= self.min_stepsize:
                self.current_iterate = x_tilde
                self.current_value = loss
                self.iterates = [x_tilde]
                self.fvalues = [loss ]
                self.P = self.generate_direction_matrix()
                self.current_idx = 0 
                self.phase = ZEXEPhase.FORWARD_DIRECTION
                self.stepsize = min(self.stepsize * self.expansion_factor, self.max_stepsize)
                if self.stepsize <= self.min_stepsize:
                    print(f"[{str(self.phase)}] MIN STEPSIZE REACHED")
                    self.h = self.h_0
                    self.k = 0
            else:
                self.stepsize = max(self.stepsize * self.contraction_factor, self.min_stepsize)
        elif self.phase == ZEXEPhase.EXPLORATION:
            print(f"[{str(self.phase)}] RECEIVED EXPLORATION VALUE")
#            current_content_size = self.manipulation_function.content_size if self.manipulation_function.content_size < 65536 else 65536
#             reg = (self.manipulation_function.how_many_sections * (8 + self.manipulation_function.content_size)) / (1000 * (8 + 65536))
#             self.de.tell(self.current_expl, loss + reg)            
# #            self.num_iters+=1
# #            if self.num_iters >= 1:#self.popsize:
#             x_tilde = (self.best[0].value / 255.0) - 0.5 # put in [-0.5, 0.5]
#             x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space
            self.current_iterate = x_tilde
            self.current_value = loss #self.best[1]
 #           self.manipulation_function.how_many_sections = self.best[2]
  #          self.manipulation_function.content_size = self.best[3]
            self.fvalues = [loss]
            self.current_idx = 0
            self.P = self.generate_direction_matrix()
            self.phase = ZEXEPhase.FORWARD_DIRECTION

#            if self.X.shape[0] > 20000:
#                self.X = self.X[10000:]
#                self.Y = self.Y[10000:]
#            exit()
        self.X = np.hstack((self.X, candidate.value.flatten()))
        self.Y = np.hstack((self.Y, loss))
#        self.X.append(x_tilde)
#        self.Y.append(loss)

    def recommend(self) -> p.Parameter:

        print("[REC] recommend: ",self.best)
        return self.best[0]






class ZEXEOptimizer(ConfiguredOptimizer):

    def __init__(self, 
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
                    seed: int = 42):
        self.name = "ZEXE"
        super().__init__(_ZEXE, locals(), as_config=True)
        self.stepsize = stepsize #if isinstance(stepsize, Callable) else lambda _: stepsize
        self.h = h #if isinstance(h, Callable) else lambda _: h
        self.max_stepsize = max_stepsize
        self.armijo_constant = armijo_constant
        self.num_directions = num_directions
        self.min_stepsize = min_stepsize
        self.contraction_factor = contraction_factor
        self.expansion_factor = expansion_factor
        self.exploration_strategy = get_expl_strategy(exploration_strategy, expl_strategy_params)
        self.seed=seed


