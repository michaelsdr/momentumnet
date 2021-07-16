# Authors: Michael Sander, Pierre Ablin
# License: MIT

"""
Original code from
Maclaurin, Dougal, David Duvenaud, and Ryan Adams.
"Gradient-based hyperparameter optimization through reversible learning."
International conference on machine learning. PMLR, 2015.
"""

import numpy as np
import torch

RADIX_SCALE = 2 ** 52


class TorchExactRep(object):
    def __init__(
        self,
        val,
        from_intrep=False,
        shape=None,
        device=None,
        from_representation=None,
    ):
        if from_representation is not None:
            intrep, store = from_representation
            self.intrep = intrep
            self.aux = BitStore(0, 0, store=store)
        else:
            if device is None:
                device = val.device.type
            if shape is not None:
                self.intrep = torch.zeros(
                    *shape, dtype=torch.long, device=device
                )
            else:
                shape = val.shape
                if from_intrep:
                    self.intrep = val
                else:
                    self.intrep = self.float_to_intrep(val)

            self.aux = BitStore(shape, device)

    def __imul__(self, a):
        self.mul(a)
        return self

    def __iadd__(self, a):
        self.add(a)
        return self

    def __isub__(self, a):
        self.sub(a)
        return self

    def __itruediv__(self, a):
        self.div(a)
        return self

    def add(self, A):
        """Reversible addition of vector or scalar A."""
        self.intrep += self.float_to_intrep(A)
        return self

    def sub(self, A):
        self.add(-A)
        return self

    def rational_mul(self, n, d):
        self.aux.push(self.intrep % d, d)  # Store remainder bits externally
        # self.intrep //= d  # Divide by denominator
        self.intrep = torch.div(self.intrep, d, rounding_mode='trunc')
        self.intrep *= n  # Multiply by numerator
        self.intrep += self.aux.pop(n)  # Pack bits into the remainder

    def mul(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(n, d)
        return self

    def div(self, a):
        n, d = self.float_to_rational(a)
        self.rational_mul(d, n)
        return self

    def float_to_rational(self, a):
        d = 2 ** 16 // int(a + 1)
        n = int(a * d + 1)
        return n, d

    def float_to_intrep(self, x):
        if type(x) is torch.Tensor:
            return (x * RADIX_SCALE).long()
        return int(x * RADIX_SCALE)

    def __repr__(self):
        return repr(self.val)

    def n_max_iter(self, beta):
        d, n = self.float_to_rational(beta)
        return int((64 - np.log2(n)) / np.abs(np.log2(n) - np.log2(d)))

    @property
    def val(self):
        return self.intrep.float() / RADIX_SCALE

    def copy(self):
        v = TorchExactRep(self.val)
        v.intrep = torch.clone(self.intrep)
        v.aux.store = torch.clone(self.aux.store)
        return v

    def reset(self):
        self.intrep.fill_(0)
        self.aux.store.fill_(0)


class BitStore(object):
    """
    Efficiently stores information with non-integer number of bits (up to 16).
    """

    def __init__(self, shape, device, store=None):
        # Use an array of Python 'long' ints which conveniently grow
        # as large as necessary. It's about 50X slower though...
        if store is not None:
            self.store = store
        else:
            self.store = torch.zeros(shape, dtype=torch.long).to(device)

    def push(self, N, M):
        """Stores integer N, given that 0 <= N < M"""
        self.store *= M
        self.store += N

    def pop(self, M):
        """Retrieves the last integer stored."""
        N = self.store % M
        # self.store //= M
        self.store = torch.div(self.store, M, rounding_mode='trunc')
        return N

    def __repr__(self):
        return repr(self.store)
