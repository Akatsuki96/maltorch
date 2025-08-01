import random
import string
from pathlib import Path
from typing import Union

import lief.PE
import numpy as np 
import torch

from maltorch.initializers.initializers import IdentityInitializer
from maltorch.manipulations.byte_manipulation import ByteManipulation


class ZEXESectionInjectionManipulation(ByteManipulation):
    def __init__(
            self,
            how_many_sections: int = 75,
            content_size: int = 512,
            only_printable_char : bool = False,
            domain_constraints=None,
            perturbation_constraints=None,
            seed : int = 123123123
    ):
        self.how_many_sections = how_many_sections
        if domain_constraints is None:
            domain_constraints = []
        if perturbation_constraints is None:
            perturbation_constraints = []
        self._sections = []
        self.content_size  = content_size
        self.only_printable_char = only_printable_char
        random.seed(seed)
        self.generator = torch.Generator().manual_seed(seed)
#        self._names = [
#            ''.join(random.choices(string.ascii_uppercase + string.digits, k=8)) for _ in range(self.how_many_sections)
#        ]
        # for i in range(how_many_sections):
        #     content = np.random.randint(0, 255, (content_size,), dtype=np.uint8)
        #     self._sections.append(content.tolist())
        super().__init__(IdentityInitializer(), domain_constraints, perturbation_constraints)

    def _apply_manipulation(
            self, x: torch.Tensor, delta: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
 #       print(x.data.cpu().flatten().tolist(), delta)
#        exit()
        # TODO: section injection is executed ONLY sample-wise, since it only works with GradFree
        lief_pe: lief.PE = lief.PE.parse(x.data.cpu().flatten().tolist())
        sections_names = delta.clone().reshape((self.how_many_sections, self.content_size + 8)).to(dtype=torch.int64)
        for i in  range(self.how_many_sections):
            #print(((sections_names[i][:8] % 95) + 33))
            name = ''.join(chr(i.item()) for i in ((sections_names[i][:8] % 95) + 33))
            content = sections_names[i][8:].tolist() if not self.only_printable_char else  ((sections_names[i][8:] % 95) + 33).tolist()
            s = lief.PE.Section(name=name)
            s.content = content#sections[i].tolist() #content[:int(len(content) * delta_i)]
            lief_pe.add_section(s)
        builder = lief.PE.Builder(lief_pe)
        builder.build()
        x = torch.atleast_2d(torch.Tensor(builder.get_build()).long())
        return x, delta

    def initialize(self, samples: torch.Tensor):
        if self.only_printable_char:
            delta = torch.randint(33, 95, (samples.shape[0], self.how_many_sections * (self.content_size + 8)), generator=self.generator, dtype=torch.int64)
        else:
            delta = torch.randint(0, 255, (samples.shape[0], self.how_many_sections * (self.content_size + 8)), generator=self.generator, dtype=torch.int64)
        return samples, delta
