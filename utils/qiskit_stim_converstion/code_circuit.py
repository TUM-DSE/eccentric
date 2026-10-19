# -*- coding: utf-8 -*-
# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

from abc import ABC, abstractmethod


class CodeCircuit(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def string2nodes(self, string, **kwargs):
        pass

    @abstractmethod
    def check_nodes(self, nodes, ignore_extras=False, minimal=False):
        pass

    @abstractmethod
    def is_cluster_neutral(self, nodes):
        return not bool(len(nodes) % 2)
