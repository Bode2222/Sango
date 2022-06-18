from Tuple1D import Tuple1DBackwards
from WFCollapse1D import WFCollapse1D

class WFCollapse1DBackContext(WFCollapse1D):
    def _getTupleObject(self, n_tiles, dims, context_dims, context_space):
        return Tuple1DBackwards(n_tiles, dims, context_dims, context_space)