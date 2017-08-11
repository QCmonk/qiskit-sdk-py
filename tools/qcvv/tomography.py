# -*- coding: utf-8 -*-

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

"""
Quantum Tomography Module

Description:
    This module contains functions for performing quantum state and quantum
    process tomography. This includes:
    - Functions for generating a set of circuits in a QuantumProgram to
      extract tomographically complete sets of measurement data.
    - Functions for generating a tomography data set from the QuantumProgram
      results after the circuits have been executed on a backend.
    - Functions for reconstructing a quantum state, or quantum process
      (Choi-matrix) from tomography data sets.

Reconstruction Methods:
    Currently implemented reconstruction methods are
    - Linear inversion by weighted least-squares fitting.
    - Fast maximum likelihood reconstruction using ref [1].

References:
    [1] J Smolin, JM Gambetta, G Smith, Phys. Rev. Lett. 108, 070502 (2012).
        Open access: arXiv:1106.5458 [quant-ph].
"""

import numpy as np
import itertools as it
import random
from functools import reduce
from re import match

from tools.qi.qi import vectorize, devectorize, outer


###############################################################
# Tomography Bases
###############################################################

class TomographyBasis(dict):
    """
    Dictionary subsclass that includes methods for adding gates to circuits.

    A TomographyBasis is a dictionary where the keys index a measurement
    and the values are a list of projectors associated to that measurement.
    It also includes two optional methods `prep_gate` and `meas_gate`:
        - `prep_gate` adds gates to a circuit to prepare the corresponding
          basis projector from an inital ground state.
        - `meas_gate` adds gates to a circuit to transform the default
          Z-measurement into a measurement in the basis.
    With the exception of built in bases, these functions do nothing unless
    they are specified by the user. They may be set by the data members
    `prep_fun` and `meas_fun`. We illustrate this with an example.
    
    Example:
        A measurement in the Pauli-X basis has two outcomes corresponding to
        the projectors:
            `Xp = [[0.5, 0.5], [0.5, 0.5]]`
            `Zm = [[0.5, -0.5], [-0.5, 0.5]]`
        We can express this as a basis by
            `BX = TomographyBasis( {'X': [Xp, Xm]} )`
        To specifiy the gates to prepare and measure in this basis we :
            ```
            def BX_prep_fun(circuit, qreg, op):
                bas, proj = op
                if bas == "X":
                    if proj == 0:
                        circuit.u2(0., np.pi, qreg)  # apply H
                    else:  # proj == 1
                        circuit.u2(np.pi, np.pi, qreg)  # apply H.X
            def BX_prep_fun(circuit, qreg, op):
                if op == "X":
                        circuit.u2(0., np.pi, qreg)  # apply H
            ```
        We can then attach these functions to the basis using:
            `BX.prep_fun = BX_prep_fun`
            `BX.meas_fun = BX_meas_fun`.
    """

    prep_fun = None
    meas_fun = None

    def prep_gate(self, circuit, qreg, op):
        """
        """
        if self.prep_fun is None:
            pass
        else:
            return self.prep_fun(circuit, qreg, op)

    def meas_gate(self, circuit, qreg, op):
        """
        """
        if self.meas_fun is None:
            pass
        else:
            return self.meas_fun(circuit, qreg, op)


def tomography_basis(basis, prep=None, meas=None):
    ret = TomographyBasis(basis)
    ret.prep_fun = prep
    ret.meas_fun = meas
    return ret


# PAULI BASIS
# This corresponds to measurements in the X, Y, Z basis where
# Outcomes 0,1 are the +1,-1 eigenstates respectively.
# State preparation is also done in the +1 and -1 eigenstates.

def __pauli_prep_gates(circuit, qreg, op):
    """
    Add state preparation gates to a circuit.
    """
    bas, proj = op
    assert(bas in ['X', 'Y', 'Z'])
    if bas == "X":
        if proj == 1:
            circuit.u2(np.pi, np.pi, qreg)  # H.X
        else:
            circuit.u2(0., np.pi, qreg)  # H
    elif bas == "Y":
        if proj == 1:
            circuit.u2(-0.5 * np.pi, np.pi, qreg)  # S.H.X
        else:
            circuit.u2(0.5 * np.pi, np.pi, qreg)  # S.H
    elif bas == "Z" and proj == 1:
        circuit.u3(np.pi, 0., np.pi, qreg)  # X


def __pauli_meas_gates(circuit, qreg,  op):
    """
    Add state measurement gates to a circuit.
    """
    assert(op in ['X', 'Y', 'Z'])
    if op == "X":
        circuit.u2(0., np.pi, qreg)  # H
    elif op == "Y":
        circuit.u2(0., 0.5 * np.pi, qreg)  # H.S^*


__PAULI_BASIS_OPS = {'X': [np.array([[0.5, 0.5],
                                    [0.5, 0.5]]),
                           np.array([[0.5, -0.5],
                                    [-0.5, 0.5]])],
                     'Y': [np.array([[0.5, -0.5j],
                                    [0.5j, 0.5]]),
                           np.array([[0.5, 0.5j],
                                    [-0.5j, 0.5]])],
                     'Z': [np.array([[1, 0],
                                    [0, 0]]),
                           np.array([[0, 0],
                                    [0, 1]])]}


# Create the actual basis
PAULI_BASIS = tomography_basis(__PAULI_BASIS_OPS,
                               prep=__pauli_prep_gates,
                               meas=__pauli_meas_gates)


# SIC-POVM BASIS
def __sic_prep_gates(circuit, qreg, op):
    """
    Add state preparation gates to a circuit.
    """
    bas, proj = op
    assert(bas == 'S')
    if bas == "S":
        theta = -2*np.arctan(np.sqrt(2))
        if proj == 1:
            circuit.u3(theta, np.pi, 0.0, qreg)
        elif proj == 2:
            circuit.u3(theta, np.pi/3, 0.0, qreg)
        elif proj == 3:
            circuit.u3(theta, -np.pi/3, 0.0, qreg)


__SIC_BASIS_OPS = {
    'S': [
        np.array([[1, 0],
                  [0, 0]]),
        np.array([[1, np.sqrt(2)],
                 [np.sqrt(2), 2]])/3,
        np.array([[1, np.exp(np.pi * 2j / 3) * np.sqrt(2)],
                  [np.exp(-np.pi * 2j / 3) * np.sqrt(2), 2]])/3,
        np.array([[1, np.exp(-np.pi * 2j / 3) * np.sqrt(2)],
                  [np.exp(np.pi * 2j / 3) * np.sqrt(2), 2]])/3
    ]}


SIC_BASIS = tomography_basis(__SIC_BASIS_OPS, prep=__sic_prep_gates)


###############################################################
# Tomography Set and labels
###############################################################

def tomography_set(qubits, meas_basis='pauli', prep_basis=None,
                   samples=None, seed=None):
    """
    """

    assert(isinstance(qubits, list))
    nq = len(qubits)

    if meas_basis == 'pauli':
        meas_basis = PAULI_BASIS

    if prep_basis == 'pauli':
        prep_basis = PAULI_BASIS
    elif prep_basis == 'SIC':
        prep_basis = SIC_BASIS

    ret = {'qubits': qubits, 'meas_basis': meas_basis}

    # add meas basis configs
    mlst = meas_basis.keys()
    meas = [dict(zip(qubits, b)) for b in it.product(mlst, repeat=nq)]
    ret['circuits'] = [{'meas': m} for m in meas]

    if prep_basis is not None:
        ret['prep_basis'] = prep_basis
        ns = len(list(prep_basis.values())[0])
        plst = [(b, s) for b in prep_basis.keys() for s in range(ns)]
        ret['circuits'] = [{'prep': dict(zip(qubits, b)), 'meas': dic['meas']} 
                           for b in it.product(plst, repeat=nq) 
                           for dic in ret['circuits']]

    if samples is None:
        return ret
    else:
        rng = random.Random(seed)
        ret['circuits'] = rng.sample(ret['circuits'], samples)
        return ret


def tomography_circuit_names(tomo_set, name=''):
    """
    """
    labels = []
    for circ in tomo_set['circuits']:
        label = ''
        # add prep
        if 'prep' in circ:
            label += '_prep_'
            for qubit, op in circ['prep'].items():
                label += '%s%d(%d)' % (op[0], op[1], qubit)
        # add meas
        label += '_meas_'
        for qubit, op in circ['meas'].items():
            label += '%s(%d)' % (op[0], qubit)
        labels.append(name+label)
    return labels


###############################################################
# Tomography circuit generation
###############################################################

def create_tomography_circuits(qp, name, qreg, creg, tomoset, silent=False):
    """
    """
    dics = tomoset['circuits']
    labels = tomography_circuit_names(tomoset, name)
    circuit = qp.get_circuit(name)

    for label, conf in zip(labels, dics):
        tmp = circuit
        # Add prep circuits
        if 'prep' in conf:
            prep = qp.create_circuit('tmp_prep', [qreg], [creg])
            for q, op in conf['prep'].items():
                tomoset['prep_basis'].prep_gate(prep, qreg[q], op)
                prep.barrier(qreg[q])
            tmp = prep + tmp
            del qp._QuantumProgram__quantum_program['tmp_prep']
        # Add measurement circuits
        meas = qp.create_circuit('tmp_meas', [qreg], [creg])
        for q, op in conf['meas'].items():
            meas.barrier(qreg[q])
            tomoset['meas_basis'].meas_gate(meas, qreg[q], op)
            meas.measure(qreg[q], creg[q])
        tmp = tmp + meas
        del qp._QuantumProgram__quantum_program['tmp_meas']
        # Add tomography circuit
        qp.add_circuit(label, tmp)

    if not silent:
        print('>> created tomography circuits for "%s"' % name)
    return labels


###############################################################
# Preformatting count data
###############################################################

def marginal_counts(counts, meas_qubits):
    """
    Compute the marginal counts for a subset of measured qubits.

    Args:
        counts (dict{str:int}): the counts returned from a backend.
        meas_qubits (list[int]): the qubits to return the marginal
                                 counts distribution for.

    Returns:
        A counts dict for the meas_qubits.abs
        Example: if counts = {'00': 10, '01': 5}
            marginal_counts(counts, [0]) returns {'0': 15, '1': 0}.
            marginal_counts(counts, [0]) returns {'0': 10, '1': 5}.
    """

    # Extract total number of qubits from count keys
    nq = len(list(counts.keys())[0])

    # keys for measured qubits only
    qs = sorted(meas_qubits, reverse=True)

    meas_keys = __counts_keys(len(qs))

    # get regex match strings for suming outcomes of other qubits
    rgx = [reduce(lambda x, y: (key[qs.index(y)] if y in qs else '\\d') + x,
                  range(nq), '')
           for key in meas_keys]

    # build the return list
    meas_counts = []
    for m in rgx:
        c = 0
        for key, val in counts.items():
            if match(m, key):
                c += val
        meas_counts.append(c)

    # return as counts dict on measured qubits only
    return dict(zip(meas_keys, meas_counts))


def __counts_keys(n):
    """Generate outcome bitstrings for n-qubits.

    Args:
        n (int): the number of qubits.

    Returns:
        A list of bitstrings ordered as follows:
        Example: n=2 returns ['00', '01', '10', '11'].
    """
    return [bin(j)[2:].zfill(n) for j in range(2 ** n)]


###############################################################
# Get results data
###############################################################

def tomography_data(results, name, tomoset):
    """
    """
    labels = tomography_circuit_names(tomoset, name)
    counts = [marginal_counts(results.get_counts(circ), tomoset['qubits'])
              for circ in labels]
    shots = [sum(c.values()) for c in counts]
    conf = tomoset['circuits']

    meas = [__meas_projector(dic['meas'], tomoset['meas_basis'])
            for dic in conf]
    if 'prep' in conf[0]:
        preps = [__prep_projector(dic['prep'], tomoset['prep_basis'])
                 for dic in conf]
        return [{'counts': c, 'shots': s, 'meas_basis': m, 'prep_basis': p}
                for c, s, m, p in zip(counts, shots, meas, preps)]
    else:
        return [{'counts': c, 'shots': s, 'meas_basis': m}
                for c, s, m in zip(counts, shots, meas)]


def __meas_projector(dic, basis):
    """
    """
    itr = it.product(*[basis[dic[i]] for i in sorted(dic.keys(), reverse=True)])
    ops = []
    for b in itr:
        ops.append(reduce(lambda acc, j: np.kron(acc, j), b, [1]))
    keys = __counts_keys(len(dic))
    return dict(zip(keys, ops))


def __prep_projector(dic, basis):
    """
    """
    ops = [dic[i] for i in sorted(dic.keys(), reverse=True)]
    ret = [1]
    for b, i in ops:
        ret = np.kron(ret, basis[b][i])
    return ret


###############################################################
# Tomographic Reconstruction functions.
###############################################################

def fit_tomography_data(data, method=None, options=None):
    """
    Reconstruct a density matrix or process-matrix from tomography data.

    If the input data is state_tomography_data the returned operator will
    be a density matrix. If the input data is process_tomography_data the
    returned operator will be a Choi-matrix in the column-vectorization
    convention.

    Args:
        data (dict): process tomography measurement data.
        method (str, optional): the fitting method to use.
            Available methods:
                - 'wizard' (default)
                - 'leastsq'
        options (dict, optional): additional options for fitting method.

    Returns:
        The fitted operator.

    Available methods:
        - 'wizard' (Default): The returned operator will be constrained to be
                              positive-semidefinite.
            Options:
            - 'trace': the trace of the returned operator.
                       The default value is 1.
            - 'beta': hedging parameter for computing frequencies from
                      zero-count data. The default value is 0.50922.
            - 'epsilon: threshold for truncating small eigenvalues to zero.
                        The default value is 0
        - 'leastsq': Fitting without postive-semidefinite constraint.
            Options:
            - 'trace': Same as for 'wizard' method.
            - 'beta': Same as for 'wizard' method.
    """
    if method is None:
        method = 'wizard'  # set default method

    if method in ['wizard', 'leastsq']:
        # get options
        trace = __get_option('trace', options)
        beta = __get_option('beta', options)
        # fit state
        rho = __leastsq_fit(data, trace=trace, beta=beta)
        if method == 'wizard':
            # Use wizard method to constrain positivity
            epsilon = __get_option('epsilon', options)
            rho = __wizard(rho, epsilon=epsilon)
        return rho
    else:
        print('error: method unknown reconstruction method "%s"' % method)


def __get_option(opt, options):
    """
    Return an optional value or None if not found.
    """
    if options is not None:
        if opt in options:
            return options[opt]
    return None


###############################################################
# Fit Method: Linear Inversion
###############################################################

def __tomo_basis_matrix(meas_basis):
    """Return a matrix of vectorized measurement operators.

    Args:
        meas_basis(list(array_like)): measurement operators [M_j].
    Returns:
        The operators S = sum_j |j><M_j|.
    """
    n = len(meas_basis)
    d = meas_basis[0].size
    S = np.array([vectorize(m).conj() for m in meas_basis])
    return S.reshape(n, d)


def __tomo_linear_inv(freqs, ops, weights=None, trace=None):
    """
    Reconstruct a matrix through linear inversion.

    Args:
        freqs (list[float]): list of observed frequences.
        ops (list[np.array]): list of corresponding projectors.
        weights (list[float] or array_like, optional):
            weights to be used for weighted fitting.
        trace (float, optional): trace of returned operator.

    Returns:
        A numpy array of the reconstructed operator.
    """
    # get weights matrix
    if weights is not None:
        W = np.array(weights)
        if W.ndim == 1:
            W = np.diag(W)

    # Get basis S matrix
    S = np.array([vectorize(m).conj()
                  for m in ops]).reshape(len(ops), ops[0].size)
    if weights is not None:
        S = np.dot(W, S)  # W.S

    # get frequencies vec
    v = np.array(freqs)  # |f>
    if weights is not None:
        v = np.dot(W, freqs)  # W.|f>
    Sdg = S.T.conj()  # S^*.W^*
    inv = np.linalg.pinv(np.dot(Sdg, S))  # (S^*.W^*.W.S)^-1

    # linear inversion of freqs
    ret = devectorize(np.dot(inv, np.dot(Sdg, v)))
    # renormalize to input trace value
    if trace is not None:
        ret = trace * ret / np.trace(ret)
    return ret


def __leastsq_fit(data, weights=None, trace=None, beta=None):
    """
    Reconstruct a state from unconstrained least-squares fitting.

    Args:
        data (list[dict]): state or process tomography data.
        weights (list or array, optional): weights to use for least squares
            fitting. The default is standard deviation from a binomial
            distribution.
        trace (float, optional): trace of returned operator. The default is 1.
        beta (float >=0, optional): hedge parameter for computing frequencies
            from zero-count data. The default value is 0.50922.

    Returns:
        A numpy array of the reconstructed operator.
    """
    if trace is None:
        trace = 1.  # default to unit trace

    ks = data[0]['counts'].keys()
    K = len(ks)
    # Get counts and shots
    ns = np.array([dat['counts'][k] for dat in data for k in ks])
    shots = np.array([dat['shots'] for dat in data for k in ks])
    # convert to freqs using beta to hedge against zero counts
    if beta is None:
        beta = 0.50922
    freqs = (ns + beta) / (shots + K * beta)

    # Use standard least squares fitting weights
    if weights is None:
        weights = np.sqrt(shots / (freqs * (1 - freqs)))

    # Get measurement basis ops
    if 'prep_basis' in data[0]:
        # process tomography fit
        ops = [np.kron(dat['prep_basis'].T, dat['meas_basis'][k])
               for dat in data for k in ks]
    else:
        # state tomography fit
        ops = [dat['meas_basis'][k] for dat in data for k in ks]

    return __tomo_linear_inv(freqs, ops, weights, trace=trace)


###############################################################
# Fit Method: Wizard
###############################################################

def __wizard(rho, epsilon=None):
    """
    Returns the nearest postitive semidefinite operator to an operator.

    This method is based on reference [1]. It constrains positivity
    by setting negative eigenvalues to zero and rescaling the positive
    eigenvalues.

    Args:
        rho (array_like): the input operator.
        epsilon(float >=0, optional): threshold for truncating small
            eigenvalues values to zero.

    Returns:
        A positive semidefinite numpy array.
    """
    if epsilon is None:
        epsilon = 0.  # default value

    dim = len(rho)
    rho_wizard = np.zeros([dim, dim])
    v, w = np.linalg.eigh(rho)  # v eigenvecrors v[0] < v[1] <...
    for j in range(dim):
        if v[j] < epsilon:
            tmp = v[j]
            v[j] = 0.
            # redistribute loop
            x = 0.
            for k in range(j + 1, dim):
                x += tmp / (dim-(j+1))
                v[k] = v[k] + tmp / (dim - (j+1))
    for j in range(dim):
        rho_wizard = rho_wizard + v[j] * outer(w[:, j])
    return rho_wizard


###############################################################
# DEPRECIATED TOMOGRAPHY API
###############################################################

def build_state_tomography_circuits(Q_program, name, qubits, qreg, creg,
                                    meas_basis='pauli', silent=False):
    """
    Add state tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is treated as a
    state preparation circuit. This function then appends the circuit with a
    tomographically overcomplete set of measurements in the Pauli basis for
    each qubit to be measured. For n-qubit tomography this result in 3 ** n
    measurement circuits being added to the quantum program.

    Args:
        Q_program (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        silent (bool, optional): hide verbose output.

    Returns:
        A list of names of the added quantum state tomography circuits.
        Example: ['circ_measX0', 'circ_measY0', 'circ_measZ0']
    """

    tomoset = tomography_set(qubits, meas_basis)
    print('WARNING: `build_state_tomography_circuits` is depreciated. ' +
          'Use `tomography_set` and `create_tomography_circuits` instead')

    return create_tomography_circuits(Q_program, name, qreg, creg, tomoset,
                                      silent=silent)


def build_process_tomography_circuits(Q_program, name, qubits, qreg, creg,
                                      prep_basis='sic', meas_basis='pauli',
                                      silent=False):
    """
    Add process tomography measurement circuits to a QuantumProgram.

    The quantum program must contain a circuit 'name', which is the circuit
    that will be reconstructed via tomographic measurements. This function
    then prepends and appends the circuit with a tomographically overcomplete
    set of preparations and measurements in the Pauli basis for
    each qubit to be measured. For n-qubit process tomography this result in
    (6 ** n) * (3 ** n) circuits being added to the quantum program:
        - 3 ** n measurements in the Pauli X, Y, Z bases.
        - 6 ** n preparations in the +1 and -1 eigenstates of X, Y, Z.

    Args:
        Q_program (QuantumProgram): A quantum program to store the circuits.
        name (string): The name of the base circuit to be appended.
        qubits (list[int]): a list of the qubit indexes of qreg to be measured.
        qreg (QuantumRegister): the quantum register containing qubits to be
                                measured.
        creg (ClassicalRegister): the classical register containing bits to
                                  store measurement outcomes.
        silent (bool, optional): hide verbose output.

    Returns:
        A list of names of the added quantum process tomography circuits.
        Example:
        ['circ_prepXp0_measX0', 'circ_prepXp0_measY0', 'circ_prepXp0_measZ0',
         'circ_prepXm0_measX0', 'circ_prepXm0_measY0', 'circ_prepXm0_measZ0',
         'circ_prepYp0_measX0', 'circ_prepYp0_measY0', 'circ_prepYp0_measZ0',
         'circ_prepYm0_measX0', 'circ_prepYm0_measY0', 'circ_prepYm0_measZ0',
         'circ_prepZp0_measX0', 'circ_prepZp0_measY0', 'circ_prepZp0_measZ0',
         'circ_prepZm0_measX0', 'circ_prepZm0_measY0', 'circ_prepZm0_measZ0']
    """

    print('WARNING: `build_process_tomography_circuits` is depreciated. ' +
          'Use `tomography_set` and `create_tomography_circuits` instead')

    tomoset = tomography_set(qubits, meas_basis, prep_basis)
    return create_tomography_circuits(Q_program, name, qreg, creg, tomoset,
                                      silent=silent)

###############################################################
# OLD Tomography circuit labels
###############################################################

def state_tomography_circuit_names(name, qubits, meas_basis='pauli'):
    """
    Return a list of state tomography circuit names.

    This list is the same as that returned by the
    build_state_tomography_circuits function.

    Args:
        name (string): the name of the original state preparation
                       circuit.
        qubits: (list[int]): the qubits being measured.

    Returns:
        A list of circuit names.
    """
    print('WARNING: `state_tomography_circuit_names` is depreciated. ' +
          'Use `tomography_set` and `tomography_circuit_names` instead')
    tomoset = tomography_set(qubits, meas_basis=meas_basis)
    return tomography_circuit_names(tomoset, name)


def process_tomography_circuit_names(name, qubits, prep_basis='sic',
                                     meas_basis='pauli'):
    """
    Return a list of process tomography circuit names.

    This list is the same as that returned by the
    build_process_tomography_circuits function.

    Args:
        name (string): the name of the original circuit to be
                       reconstructed.
        qubits: (list[int]): the qubits being measured.

    Returns:
        A list of circuit names.
    """
    print('WARNING: `process_tomography_circuit_names` is depreciated.' +
          'Use `tomography_set` and `tomography_circuit_names` instead')
    tomoset = tomography_set(qubits, meas_basis=meas_basis,
                             prep_basis=prep_basis)
    return tomography_circuit_names(tomoset, name)


###############################################################
# OLD Getting outcome basis operators
###############################################################

def state_tomography_data(Q_result, name, meas_qubits, meas_basis='pauli'):
    """
    Return a list of state tomography measurement outcomes.

    Args:
        Q_result (Result): Results from execution of a state tomography
            circuits on a backend.
        name (string): The name of the base state preparation circuit.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        meas_basis (basis dict, optional): the basis used for measurement. Default
            is the Pauli basis.

    Returns:
        A list of dicts for the outcome of each state tomography
        measurement circuit. The keys of the dictionary are
        {
            'counts': dict('str': int),
                      <the marginal counts for measured qubits>,
            'shots': int,
                     <total number of shots for measurement circuit>
            'meas_basis': dict('str': np.array)
                          <the projector for the measurement outcomes>
        }
    """
    print('WARNING: `state_tomography_data` is depreciated. ' +
          'Use `tomography_set` and `tomography_data` instead')
    tomoset = tomography_set(meas_qubits, meas_basis=meas_basis)
    return tomography_data(Q_result, name, tomoset)


def process_tomography_data(Q_result, name, meas_qubits, prep_basis='sic',
                            meas_basis='pauli'):
    """
    Return a list of process tomography measurement outcomes.

    Args:
        Q_result (Result): Results from execution of a process tomography
            circuits on a backend.
        name (string): The name of the circuit being reconstructed.
        meas_qubits (list[int]): a list of the qubit indexes measured.
        basis (basis dict, optional): the basis used for measurement. Default
            is the Pauli basis.

    Returns:
        A list of dicts for the outcome of each process tomography
        measurement circuit. The keys of the dictionary are
        {
            'counts': dict('str': int),
                      <the marginal counts for measured qubits>,
            'shots': int,
                     <total number of shots for measurement circuit>
            'meas_basis': dict('str': np.array),
                          <the projector for the measurement outcomes>
            'prep_basis': np.array,
                          <the projector for the prepared input state>
        }
    """
    print('WARNING: `process_tomography_data` is depreciated. ' +
          'Use `tomography_set` and `tomography_data` instead')
    tomoset = tomography_set(meas_qubits, meas_basis=meas_basis,
                             prep_basis=prep_basis)
    return tomography_data(Q_result, name, tomoset)
