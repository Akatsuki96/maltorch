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







class _DZEXE(Optimizer): # It has to extend optimizer (ConfiguredOptimizer or Optimizer?) class of nevergrad?


    def __init__(self, parametrization, 
                    budget: tp.Optional[int] = None, 
                    num_workers: int = 1,
                    *,
                    config):
        super().__init__(parametrization, budget=budget, num_workers=num_workers)

        self._config = ZEXEOptimizer() if config is None else config
        self.num_directions = self._config.num_directions
        self.max_num_sections = self._config.max_num_sections
        self.max_content_size = self._config.max_content_size
        self.popsize = self._config.popsize
        self.de = _DE(p.Array(shape=(3, )).set_bounds(0, 1), config=DifferentialEvolution(popsize=self.popsize, crossover="twopoints"))
        self.de_initialized = False
        self.manipulation_function= self._config.manipulation_function
        self.no_exploit = self._config.no_exploit
        self.max_pert_size = self.max_num_sections * (8 + self.max_content_size)  #1000 * (8 + 32768)#65536)
        self.num_sections = self.manipulation_function.how_many_sections #self._config.num_sections
        self.armijo_constant = self._config.armijo_constant
        self.current_iterate = None
        self.max_ls_steps = self._config.max_ls_steps
        self.num_ls_steps = 0
        self.stepsize=1
        self.ls_values = []
        self.ls_iterates = []
        self.phase = ZEXEPhase.ITERATE
        self.rnd_state = np.random.RandomState(self._config.seed)
        self.current_idx = 0
        self.d = self.parametrization.value.shape[1]
        self.best = None
        self.k = 0
        self.g = None
        self.X, self.Y = np.empty((0,), dtype=np.int64), np.empty((0,), dtype=np.int64)# [], []

    def generate_direction_matrix(self):
        return self.rnd_state.choice([-1.0, 0.0, 1.0], (self.d, self.num_directions) ) #2 * (self.rnd_state.rand(self.d, self.num_directions) < 0.5) - 1
        # v = self.rnd_state.randn(self.d)
        # v /= np.linalg.norm(v, ord=2)
        # indices = self.rnd_state.choice(self.d, self.num_directions, replace=False)
        # I = np.zeros((self.d, self.num_directions))
        # I[indices, range(self.num_directions)] = 1.0
        # H = I - 2 * np.outer(v, v[indices])
        # return H


    def _grad_ask(self):
        g = np.zeros(self.d)
        for i in range(self.num_directions):
            g += (self.fvalues[i + 1] - self.fvalues[0])  * self.P[:, i]
        return np.sign(g)

    def _get_exploration_value(self):
        self.current_expl = self.de.ask()
        pert_size = int((self.max_pert_size - 512)* self.current_expl.value[0] + 512)

#        c_size = int((pert_size - 1024)* self.current_expl.value[1] + 1024)
#        print("PERT SIZE: ", pert_size, c_size)
        n_sec =  int((self.max_num_sections - 1)* self.current_expl.value[1] + 1)
        c_size = (pert_size - 8) // n_sec
#        n_sec = int(pert_size // (c_size - 8))
#        if c_size < 1024:
#            c_size = 1024
        # n_sec = int((self.max_sections - 1) * self.current_expl.value[0] + 1)
  #       c_size = int((self.max_content_size - 512)* self.current_expl.value[1] + 512)
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
        self.manipulation_function.how_many_sections = n_sec
        self.manipulation_function.content_size = c_size #- 8
        self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)
        self.d = new_iterate.shape[0]
        return self.parametrization

    def _internal_ask_candidate(self) -> p.Parameter:
        if self.phase == ZEXEPhase.FORWARD_DIRECTION:
            new_iterate = (self.current_iterate + self.h[self.current_idx] * self.P[:, self.current_idx].reshape((1, self.d))) % 256
            
        if self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            new_iterate = (self.current_iterate  - self.stepsize * self.h * self.g ) % 256

        if self.phase == ZEXEPhase.EXPLORATION:
            return self._get_exploration_value()


        if self.phase == ZEXEPhase.ITERATE:
            self.g = self._grad_ask()
            self.k += 1

            if self.no_exploit or ( self.best is None or self.best[1] > -0.5): #np.linalg.norm(self.g) < 0.01: 
                print("EXPLORATION")
                self.phase = ZEXEPhase.EXPLORATION
#                self.num_iters = 0
                return self._get_exploration_value()# self.parametrization#new_parametrization#self.parametrization.spawn_child(new_iterate.reshape(1, -1).astype(np.int64)) 
            else:
                print("EXPLOITATION")
                self.phase = ZEXEPhase.LINE_SEARCH_STEP
#                candidate = self.best[0]

                self.current_iterate = self.best[0].value
                self.current_value = self.best[1]
                self.manipulation_function.how_many_sections = self.best[2]
                self.manipulation_function.content_size = self.best[3]
                new_iterate = (self.current_iterate - self.stepsize* self.h *  self.g) % 256
        self.parametrization= p.Array(init=new_iterate).set_bounds(0, 255)

        return self.parametrization#.spawn_child(new_iterate.astype(np.int64)) 


    def _internal_tell_candidate(self, candidate: p.Parameter, loss: tp.FloatLoss) -> None:
#        x_tilde = (candidate.value / 255.0) - 0.5 # put in [-0.5, 0.5]
#        x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space
        x_tilde = candidate.value
        if self.best is None or (self.best[1] > -0.5 and self.best[1] > loss):
            self.best = (candidate, loss, self.manipulation_function.how_many_sections, self.manipulation_function.content_size)


 #       if self.reduce_size:
        if (loss < -0.5 and ( (self.best[2] * (8 + self.best[3]) > self.manipulation_function.how_many_sections * (8+ self.manipulation_function.content_size) ) or (self.best[2] * (8 + self.best[3]) == self.manipulation_function.how_many_sections * (8+ self.manipulation_function.content_size) and self.best[1] > loss) )):
            self.best = (candidate, loss, self.manipulation_function.how_many_sections, self.manipulation_function.content_size)
            self.max_pert_size = self.manipulation_function.how_many_sections * (8+ self.manipulation_function.content_size)
 
 
        if self.g is not None:
            print(f"[--] k: {self.k}\tf(x_k): {loss}\tx_k: {candidate.value}\t||g||: {np.linalg.norm(self.g)}\tbest: {self.best[1]}\td: {self.best[2] * (8 + self.best[3])}")

        if self.current_iterate is None:
            self.current_iterate = x_tilde #candidate.value
        if self.phase == ZEXEPhase.ITERATE:
            self.phase = ZEXEPhase.FORWARD_DIRECTION
            self.iterates = [x_tilde]
            self.fvalues = [loss]
            self.ls_iterates, self.ls_values = [], []
            self.current_idx = 0
            self.stepsize = 1
            self.P = self.generate_direction_matrix()
            self.h = self.rnd_state.randint(1, 256, self.P.shape[1])
        elif self.phase == ZEXEPhase.FORWARD_DIRECTION:
            self.iterates.append(x_tilde)
            self.fvalues.append(loss)
            self.current_idx += 1
            if self.current_idx == self.num_directions:
                self.phase = ZEXEPhase.ITERATE
                self.current_idx = 0
        elif self.phase == ZEXEPhase.LINE_SEARCH_STEP:
            print("[LS] phase: {},  loss: {}, cond: {}, stepsize: {}".format(self.phase, loss, self.current_value - self.armijo_constant, self.stepsize))
            #cond = self.current_value - self.armijo_constant * self.stepsize * np.square(np.linalg.norm(self.g))
            self.ls_values.append(loss)
            self.ls_iterates.append(x_tilde)
            if loss < self.current_value - self.armijo_constant:# * self.stepsize * np.square(np.linalg.norm(self.g)):
                print("[--] CONDITION STATIFIED -> BUILD DIRECTION")
                self.current_iterate = x_tilde
                self.current_value = loss
                self.iterates = [x_tilde]
                self.fvalues = [loss]
                self.P = self.generate_direction_matrix()
                self.h = self.rnd_state.randint(1, 256, self.P.shape[1])
                self.current_idx = 0 
                self.phase = ZEXEPhase.FORWARD_DIRECTION
                self.stepsize = 1 #min(self.stepsize * self.expansion_factor, self.max_stepsize)
                self.ls_iterates = []
                self.ls_values = []
                self.num_ls_steps = 0
            elif self.num_ls_steps >= self.max_ls_steps:
                print("[--] MAX LS STEPS PERFORMED -> EXPLORATION")
                x_tilde = self.ls_iterates[np.argmin(self.ls_values)]
                loss = self.ls_values[np.argmin(self.ls_values)]
                self.current_iterate = x_tilde
                self.current_value = loss
                self.iterates = [x_tilde]
                self.fvalues = [loss]
                self.ls_iterates = []
                self.ls_values = []
                self.P = self.generate_direction_matrix()
                self.h = self.rnd_state.randint(1, 256, self.P.shape[1])
                self.phase = ZEXEPhase.FORWARD_DIRECTION
                self.current_idx = 0
                self.stepsize = 1
                self.num_ls_steps = 0
            else:
                self.stepsize = (self.stepsize  + 1) % 256 #* self.contraction_factor
                self.num_ls_steps += 1
        elif self.phase == ZEXEPhase.EXPLORATION:
            print("[--] RECEIVED EXPLORATION VALUE")
#            current_content_size = self.manipulation_function.content_size if self.manipulation_function.content_size < 65536 else 65536
#            reg = (self.manipulation_function.how_many_sections * (8 + self.manipulation_function.content_size)) / (1000 * (8 + 65536))
            self.de.tell(self.current_expl, loss)            
#            self.num_iters+=1
#            if self.num_iters >= 1:#self.popsize:
            x_tilde = self.best[0].value#(self.best[0].value / 255.0) - 0.5 # put in [-0.5, 0.5]
#            x_tilde = np.atanh(1.999999 * x_tilde) # go to tanh space
            self.current_iterate = x_tilde
            self.current_value = self.best[1]
            self.manipulation_function.how_many_sections = self.best[2]
            self.manipulation_function.content_size = self.best[3]
            self.fvalues = [self.best[1]]
            self.d = (self.manipulation_function.how_many_sections * (8 + self.manipulation_function.content_size))
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
        self.manipulation_function.how_many_sections = self.best[2]
        self.manipulation_function.content_size = self.best[3]
        return self.best[0]






class DZEXEOptimizer(ConfiguredOptimizer):

    def __init__(self, 
                    num_directions : int = 1,
                    manipulation_function = None,
                    armijo_constant : float = 1e-5,
                    max_num_sections : int = 100,
                    max_content_size : int = 65536,
                    no_exploit =False,
                    popsize : int = 10,
                    max_ls_steps = 10,
                    seed: int = 42):
        self.name = "DZEXE"
        super().__init__(_DZEXE, locals(), as_config=True)
        self.popsize = popsize
        self.max_ls_steps = max_ls_steps
        self.no_exploit = no_exploit
        self.max_num_sections = max_num_sections
        self.max_content_size = max_content_size
        self.armijo_constant = armijo_constant
        self.num_directions = num_directions
        self.manipulation_function= manipulation_function
        self.seed=seed


