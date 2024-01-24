from __future__ import annotations

from e2cnn.group import Group
from e2cnn.group import IrreducibleRepresentation, Representation
from e2cnn.group import utils

from .cyclicgroup import CyclicGroup

import numpy as np

from typing import Tuple, Callable, Iterable, List

__all__ = ["SO2"]

_cached_group_instance = None


class SO2(Group):

    def __init__(self, maximum_frequency: int):
        r"""
       Build an instance of the special orthogonal group :math:`SO(2)` which contains continuous planar rotations.
        
        A group element is a rotation :math:`r_\theta` of :math:`\theta \in [0, 2\pi)`, with group law
        :math:`r_\alpha \cdot r_\beta = r_{\alpha + \beta}`.
        
        Elements are implemented as floating point numbers :math:`\theta \in [0, 2\pi)`.
        
        .. note ::
            Since the group has infinitely many irreducible representations, it is not possible to build all of them.
            Each irrep is associated to one unique frequency and the parameter ``maximum_frequency`` specifies
            the maximum frequency of the irreps to build.
            New irreps (associated to higher frequencies) can be manually created by calling the method
            :meth:`~e2cnn.group.SO2.irrep` (see the method's documentation).
        
        Args:
            maximum_frequency (int): the maximum frequency to consider when building the irreps of the group
        
        """
        
        assert (isinstance(maximum_frequency, int) and maximum_frequency >= 0)
        
        super(SO2, self).__init__("SO(2)", True, True)
        
        self._maximum_frequency = maximum_frequency
        
        self.identity = 0.
        
        self._build_representations()
        
    def inverse(self, element: float) -> float:
        r"""
        Return the inverse element of the input element: given an angle, the method returns its opposite

        Args:
            element (float): an angle :math:`\theta`

        Returns:
            its opposite :math:`-\theta \mod 2\pi`
        """
        
        return (-element) % (2*np.pi)

    def combine(self, e1: float, e2: float) -> float:
        r"""
        Return the sum of the two input elements: given two angles, the method returns their sum

        Args:
            e1 (float): an angle :math:`\alpha`
            e2 (float): another angle :math:`\beta`

        Returns:
            their sum :math:`(\alpha + \beta) \mod 2\pi`
            
        """
        # return e1 + e2
        return (e1 + e2) % (2.*np.pi)

    def equal(self, e1: float, e2: float) -> bool:
        r"""
        
        Check if the two input values corresponds to the same element, i.e. the same angle.
        
        The method accounts for a small absolute and relative tollerance and for the cyclicity of the the angles,
        i.e. the fact that :math:`0 + \epsilon \simeq 2\pi - \epsilon` for small :math:`\epsilon`
        
        Args:
            e1 (float): an angle :math:`\alpha`
            e2 (float): another angle :math:`\beta`

        Returns:
            whether the two angles are the same, i.e. if :math:`\beta \simeq \alpha \mod 2\pi`

        """
        return utils.cycle_isclose(e1, e2, 2 * np.pi)

    def is_element(self, element: float) -> bool:
        return isinstance(element, float)

    def testing_elements(self) -> Iterable[float]:
        r"""
        A finite number of group elements to use for testing.
        """
        N = 4*13
        return iter([i * 2. * np.pi / N for i in range(N)])
    
    def __eq__(self, other):
        if not isinstance(other, SO2):
            return False
        else:
            return self.name == other.name and self._maximum_frequency == other._maximum_frequency

    def subgroup(self, id: int) -> Tuple[Group, Callable, Callable]:
        r"""
        Restrict the current group to the cyclic subgroup :math:`C_M` with order :math:`M` which is generated
        by :math:`r_{\frac{2\pi}{M}}`.
        
        The method takes as input the integer :math:`M` identifying of the subgroup to build
        (the order of the subgroup).
        
        Args:
            id (int): the integer :math:`M` identifying the subgroup

        Returns:
            a tuple containing

                - the subgroup,

                - a function which maps an element of the subgroup to its inclusion in the original group and

                - a function which maps an element of the original group to the corresponding element in the subgroup (returns None if the element is not contained in the subgroup)
              
        """

        assert isinstance(id, int)

        order = id
    
        # Build the subgroup

        if id not in self._subgroups:
            # take the elements of the group generated by "2pi/order"
            sg = CyclicGroup(order)
            parent_mapping = lambda e, order=order: e*2*np.pi/order
            # child_mapping = lambda e, order=order: None if divmod(e, 2.*np.pi/order)[1] > 1e-15 else int(round(e * order / (2.*np.pi)))
            child_mapping = lambda e, order=order: None if not utils.cycle_isclose(e, 0., 2.*np.pi/order) else int(round(e * order / (2.*np.pi)))

            self._subgroups[id] = sg, parent_mapping, child_mapping
    
        return self._subgroups[id]

    def _restrict_irrep(self, irrep: str, id: int) -> Tuple[np.matrix, List[str]]:
        r"""
        
        Restrict the input irrep to the subgroup :math:`C_M` with order "M".
        More precisely, it restricts to the subgroup generated by :math:`2 \pi /order`.
        
        The method takes as input the integer :math:`M` identifying of the subgroup to build (the order of the subgroup)

        Args:
            irrep (str): the name/identifier of the irrep to restrict
            id (int): the integer :math:`M` identifying the subgroup

        Returns:
            a pair containing the change of basis and the list of irreps of the subgroup which appear in the restricted irrep
            
        """

        irr = self.irreps[irrep]
    
        # Build the subgroup
        sg, _, _ = self.subgroup(id)
    
        order = id
    
        change_of_basis = None
        irreps = []
        
        f = irr.attributes["frequency"] % order
    
        if f > order/2:
            f = order - f
            change_of_basis = np.array([[1, 0], [0, -1]])
        else:
            change_of_basis = np.eye(irr.size)
    
        r = f"irrep_{f}"
    
        irreps.append(r)
        if sg.irreps[r].size < irr.size:
            irreps.append(r)
        
        return change_of_basis, irreps
    
    def _build_representations(self):
        r"""
        Build the irreps for this group

        """
    
        # Build all the Irreducible Representations
    
        k = 0
    
        # add Trivial representation
        self.irrep(k)
    
        for k in range(self._maximum_frequency + 1):
            self.irrep(k)
    
        # Build all Representations
    
        # add all the irreps to the set of representations already built for this group
        self.representations.update(**self.irreps)

    @property
    def trivial_representation(self) -> Representation:
        return self.representations['irrep_0']
    
    def irrep(self, k: int) -> IrreducibleRepresentation:
        r"""
        Build the irrep with rotational frequency :math:`k` of :math:`SO(2)`.
        Notice: the frequency has to be a non-negative integer.
        
        Args:
            k (int): the frequency of the irrep

        Returns:
            the corresponding irrep

        """
    
        assert k >= 0
    
        name = f"irrep_{k}"
    
        if name not in self.irreps:

            if k == 0:
                # Trivial representation
                irrep = lambda element, identity=np.eye(1): identity
                character = lambda e: 1
                supported_nonlinearities = ['pointwise', 'norm', 'gated', 'gate']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 1, 1,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              # trivial=True,
                                                              frequency=0
                                                              )
            else:

                # 2 dimensional Irreducible Representations
        
                # build the rotation matrix with rotation order 'k'
                irrep = lambda element, k=k: utils.psi(element, k=k)
        
                # build the trace of this matrix
                character = lambda element, k=k: np.cos(k * element) + np.cos(k * element)
                supported_nonlinearities = ['norm', 'gated']
                self.irreps[name] = IrreducibleRepresentation(self, name, irrep, 2, 2,
                                                              supported_nonlinearities=supported_nonlinearities,
                                                              character=character,
                                                              frequency=k)

        return self.irreps[name]

    @staticmethod
    def _generator(maximum_frequency: int = 10) -> 'SO2':
        global _cached_group_instance
        if _cached_group_instance is None:
            _cached_group_instance = SO2(maximum_frequency)
        elif _cached_group_instance._maximum_frequency < maximum_frequency:
            _cached_group_instance._maximum_frequency = maximum_frequency
            _cached_group_instance._build_representations()
    
        return _cached_group_instance

