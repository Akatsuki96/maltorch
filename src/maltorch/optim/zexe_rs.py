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


class _ZEXERS(Optimizer): # It has to extend optimizer (ConfiguredOptimizer or Optimizer?) class of nevergrad?


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
        self.expl_strategy = self._config.expl_strategy
        self.h_0 = self._config.h
        self.manipulation_function= self._config.manipulation_function
        self.max_pert_size = self.manipulation_function.how_many_sections * (8 + self.manipulation_function.content_size)  #1000 * (8 + 32768)#65536)
        self.num_sections = self.manipulation_function.how_many_sections #self._config.num_sections
        self.popsize = self._config.popsize
        self.de = _DE(p.Array(shape=(3, )).set_bounds(0, 1), config=DifferentialEvolution(popsize=self.popsize, crossover="twopoints"))

       # self.content_size = self._config.content_size

        self.reduce_size = self._config.reduce_size
#        self.exploration_strategy = AddSection(self.num_section_to_add, self.manipulation_function.content_size)
        self.armijo_constant = self._config.armijo_constant
        self.min_stepsize = self._config.min_stepsize
        self.max_stepsize = self._config.max_stepsize
        self.contraction_factor = self._config.contraction_factor
        self.expansion_factor = self._config.expansion_factor
        self.current_iterate = None
        self.phase = ZEXEPhase.ITERATE
        self.beta = self._config.beta 
        self.rnd_state = np.random.RandomState(self._config.seed)
        self.current_idx = 0
        self.d = self.parametrization.value.shape[1]
        self.best = None
        self.best_light = None
        self.k = 0
        self.g = None
        self.X, self.Y = np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)# [], []

    def generate_direction_matrix(self):
#        return self.rnd_state.choice([-1.0, 0.0, 1.0], (self.d, self.num_directions) ) #2 * (self.rnd_state.rand(self.d, self.num_directions) < 0.5) - 1
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

    def _get_exploration_value(self):
        if self.expl_strategy == 'ga':
            self.current_expl = self.de.ask()
            n_sec = self.manipulation_function.how_many_sections
            c_size = self.manipulation_function.content_size
            st_from = int(self.X.shape[0] *self.current_expl.value[2])
            total_size = (c_size +8 )*n_sec
            new_iterate = np.empty((0,), dtype=np.int64)
            print("[EXPLORATION] NUM SEC: {n_sec}".format(n_sec=n_sec))
            print("[EXPLORATION] CONTENT SIZE: {c_size}".format(c_size=c_size))
            while st_from + total_size - new_iterate.shape[0] > self.X.shape[0]:
                new_iterate = np.hstack((new_iterate, self.X[st_from:]))
                st_from = 0
            if new_iterate.shape[0] < total_size:
                new_iterate = np.hstack((new_iterate, self.X[st_from:st_from + total_size - new_iterate.shape[0]]))
#            self.manipulation_function.how_many_sections = n_sec
#            self.manipulation_function.content_size = c_size #- 8
            self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)
            self.d = new_iterate.shape[0]
            return self.parametrization
        return self.parametrization.sample().set_integer_casting()

    def _internal_ask_candidate(self) -> p.Parameter:
        if self.phase == ZEXEPhase.FORWARD_DIRECTION:
            print(self.current_iterate.shape, self.d, self.P[:, self.current_idx].shape)
            new_iterate = self.current_iterate + self.h * self.P[:, self.current_idx].reshape((1, self.d))
            
        if self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            new_iterate = self.current_iterate  - self.stepsize * self.g 

        if self.phase == ZEXEPhase.EXPLORATION:

            if self.expl_strategy == 'langevin':
                z = self.beta(self.k) * self.rnd_state.randn(self.d)
                new_iterate = self.current_iterate - self.stepsize * self.g + z
                new_iterate = (255.0*((np.tanh(new_iterate)/2) + 0.5)).astype(np.int64)
                self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)
                return self.parametrization

            return self._get_exploration_value()


        if self.phase == ZEXEPhase.ITERATE:
            self.g = self._grad_ask()
#            self.current_value = self.fvalues[0] # f(x_k)
            self.k += 1
            print("VALUES: ", self.fvalues)
            self.h = max(self.h / self.k, 0.01)

            if np.linalg.norm(self.g) < 0.1: 
                print("EXPLORATION")
                print("EXPL STRATEGY: ", self.expl_strategy)
                self.phase = ZEXEPhase.EXPLORATION

                if self.expl_strategy == 'langevin':
                    z = self.beta(self.k) * self.rnd_state.randn(self.d)
                    new_iterate = self.current_iterate - self.stepsize * self.g + z
                    new_iterate = (255.0*((np.tanh(new_iterate)/2) + 0.5)).astype(np.int64)
                    self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)

                    return self.parametrization
                return self._get_exploration_value()# self.parametrization#new_parametrization#self.parametrization.spawn_child(new_iterate.reshape(1, -1).astype(np.int64)) 
            else:
                print("EXPLOITATION")
                self.phase = ZEXEPhase.LINE_SEARCH_STEP
                candidate = self.best[0]
                x_tilde = (candidate.value / 255.0) - 0.5 # put in [-0.5, 0.5]
                x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space

                self.current_iterate = x_tilde
                self.current_value = self.best[1]
                z = 0.0
                if self.expl_strategy == 'langevin':
                    z = self.beta(self.k) * self.rnd_state.randn(self.d)
                new_iterate = self.current_iterate - self.stepsize * self.g + z
        
        new_iterate = (255.0*((np.tanh(new_iterate)/2) + 0.5)).astype(np.int64)
        self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)

        return self.parametrization#.spawn_child(new_iterate.astype(np.int64)) 


    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
        x_tilde = (candidate.value / 255.0) - 0.5 # put in [-0.5, 0.5]
        x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space

        if self.best is None or (self.best[1] > -0.5 and self.best[1] > loss):
            self.best = (candidate, loss, self.manipulation_function.how_many_sections, self.manipulation_function.content_size)

        if loss < self.best[1]:
            self.best = (candidate, loss, self.manipulation_function.how_many_sections, self.manipulation_function.content_size)

        if self.g is not None:
            print(f"[--] k: {self.k}\tf(x_k): {loss}\tx_k: {candidate.value}\th: {self.h}\t||g||: {np.linalg.norm(self.g)}\tbest: {self.best[1]}\td: {self.best[2] * (8 + self.best[3])}")

        if self.current_iterate is None:
            self.current_iterate = x_tilde #candidate.value
        if self.phase == ZEXEPhase.ITERATE:
            self.phase = ZEXEPhase.FORWARD_DIRECTION
            self.iterates = [x_tilde]
            self.fvalues = [loss]
            self.current_idx = 0
            self.P = self.generate_direction_matrix()
        elif self.phase == ZEXEPhase.FORWARD_DIRECTION:
            self.iterates.append(x_tilde)
            self.fvalues.append(loss)
            self.current_idx += 1
            if self.current_idx == self.num_directions:
                self.phase = ZEXEPhase.ITERATE
                self.current_idx = 0
        elif self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            print("[LS] phase: {},  loss: {}, cond: {}, stepsize: {}".format(self.phase, loss, self.current_value - self.armijo_constant * self.stepsize * np.square(np.linalg.norm(self.g)), self.stepsize))
            cond = self.current_value - self.armijo_constant * self.stepsize * np.square(np.linalg.norm(self.g))
            if loss < self.current_value - self.armijo_constant * self.stepsize * np.square(np.linalg.norm(self.g)):
                print("[--] CONDITION STATIFIED -> BUILD DIRECTION")
                self.current_iterate = x_tilde
                self.current_value = loss
                self.iterates = [x_tilde]
                self.fvalues = [loss]
                self.P = self.generate_direction_matrix()
                self.current_idx = 0 
                self.phase = ZEXEPhase.FORWARD_DIRECTION
                self.stepsize = min(self.stepsize * self.expansion_factor, self.max_stepsize)
            elif self.stepsize <= self.min_stepsize:
                print("[--] MIN STEPSIZE REACHED -> EXPLORATION")
                self.current_iterate = x_tilde
                self.current_value = loss
                self.iterates = [x_tilde]
                self.fvalues = [loss]
                self.P = self.generate_direction_matrix()
                self.phase = ZEXEPhase.FORWARD_DIRECTION
                self.current_idx = 0
                self.stepsize = self.init_stepsize
                self.k = 0
                self.h = self.h_0
            else:
                self.stepsize = self.stepsize * self.contraction_factor
        elif self.phase == ZEXEPhase.EXPLORATION:
            if self.expl_strategy == 'ga':
                self.de.tell(self.current_expl, loss)

            x_tilde = (self.best[0].value / 255.0) - 0.5 # put in [-0.5, 0.5]
            x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space
            self.current_iterate = x_tilde
            self.current_value = self.best[1]
            self.fvalues = [self.best[1]]
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






class ZEXERSOptimizer(ConfiguredOptimizer):

    def __init__(self, 
                    stepsize: float = 1.0, 
                    h : float = 1.0,
                    num_directions : int = 1,
                    manipulation_function = None,
                    armijo_constant : float = 1e-5,
                    min_stepsize : float = 1e-5,
                    max_stepsize : float = 1e3,
                    popsize : int = 10,
                    contraction_factor : float = 0.5,
                    expansion_factor : float = 2.0,
                    reduce_size : bool = False,
                    expl_strategy  = 'rs',                    
                    beta : float = lambda k : 0.001 * np.log(k + 1) / (k + 1),
                    seed: int = 42):
        self.name = "ZEXERS"
        super().__init__(_ZEXERS, locals(), as_config=True)
        self.stepsize = stepsize #if isinstance(stepsize, Callable) else lambda _: stepsize
        self.h = h #if isinstance(h, Callable) else lambda _: h
        self.popsize = popsize
        self.max_stepsize = max_stepsize
        self.armijo_constant = armijo_constant
        self.num_directions = num_directions
        self.min_stepsize = min_stepsize
        self.reduce_size = reduce_size
        self.manipulation_function= manipulation_function
        self.contraction_factor = contraction_factor
        self.expansion_factor = expansion_factor
        self.expl_strategy = expl_strategy
        self.beta = beta
        self.seed=seed


