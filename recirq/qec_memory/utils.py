# Copyright 2026 Google
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
import scipy
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal


def get_pre_loop_key(keys: list[str]) -> int:
    """Find the last measure key before the ones from within the CircuitOperation.

    Args:
        keys: A list of the measure keys.

    Returns:
        The key as an integer.
    """
    digit_keys = sorted([int(key) for key in keys if key.isdigit()])
    for i in range(len(digit_keys) - 1):
        if digit_keys[i + 1] - digit_keys[i] > 1:
            return digit_keys[i]


def replace_loop_repetitions(
    circuit: cirq.Circuit, cycles: int, original_cycles: int = 10
) -> cirq.Circuit:
    """Replace the number of times a CircuitOperation is repeated within a circuit.

    Args:
        circuit: A circuit containing a single CircuitOperation.
        cycles: The desired number of QEC cycles.
        original_cycles: The number of QEC cycles in the input circuit.

    Returns:
        A modified circuit.
    """
    new_moments = []
    for moment in circuit:
        if (
            len(moment.operations) == 1
            and type(moment.operations[0]) == cirq.CircuitOperation
        ):
            op = moment.operations[0]
            current_repetitions = op.repetitions
            new_moments.append(
                cirq.Moment(
                    cirq.CircuitOperation(
                        circuit=op.circuit,
                        repetitions=cycles - original_cycles + op.repetitions,
                        qubit_map=op.qubit_map,
                        measurement_key_map=op.measurement_key_map,
                        param_resolver=op.param_resolver,
                        parent_path=op.parent_path,
                        repeat_until=op.repeat_until,
                    )
                )
            )
        else:
            new_moments.append(moment)
    return cirq.Circuit.from_moments(*new_moments)


def fit_logical_error_per_cycle(
    cycles: np.ndarray, lep: np.ndarray, d_lep: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the logical error per cycle.

    See Section III of the SM to Nature 614, 676–681 (2023).

    Args:
        cycles: The cycles for which the lep is measured.
        lep: The logical error probability.

    Returns:
        The parameters A and epsilon in the fit and their covariance matrix.
    """
    fit_function = lambda t, amp, epsilon: amp * (1 - 2 * epsilon) ** t
    fidelity = 1 - 2 * lep
    d_fidelity = 2 * d_lep
    fit = scipy.optimize.curve_fit(
        fit_function,
        cycles,
        fidelity,
        sigma=d_fidelity,
        absolute_sigma=True,
        p0=(1.0, 0.01),
    )
    return fit


def identify_measure_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify which qubits are used for mid-circuit measurements.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The measure qubits.
    """
    measure_qubits = set()
    for op in circuit[:-1].all_operations():
        if cirq.is_measurement(op) and not type(op) == cirq.CircuitOperation:
            for q in op.qubits:
                measure_qubits.add(q)
    return measure_qubits


def identify_terminal_measured_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify all of the qubits that are measured in the final moment.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The qubits that are measured terminally.
    """
    measure_qubits = set()
    for op in circuit[-1].operations:
        if cirq.is_measurement(op) and not type(op) == cirq.CircuitOperation:
            for q in op.qubits:
                measure_qubits.add(q)
    return measure_qubits


def identify_data_qubits(circuit: cirq.Circuit) -> set[cirq.GridQubit]:
    """Identify the data qubits in a QEC circuit.

    Args:
        circuit: The circuit to inspect.

    Returns:
        The data qubits
    """
    return identify_terminal_measured_qubits(circuit).difference(
        identify_measure_qubits(circuit)
    )


def add_sweep_bits(
    circuit: cirq.Circuit, rng: np.random.Generator = np.random.default_rng()
) -> cirq.Circuit:
    """Add random X gates on data qubits at the beginning of a QEC circuit.

    If the first moment contains a single wait gate, then insert the X gates after it.

    Args:
        circuit: The circuit to add the gates to.
        rng: A pseudorandom number generator.

    Returns:
        The modified circuit.
    """
    data_qubits = list(identify_data_qubits(circuit))
    include = rng.random(len(data_qubits))
    qubits_to_flip = np.array(data_qubits)[include > 0.5]
    idx = int(type(circuit[0].operations[0].gate) == cirq.WaitGate)
    return circuit[:idx] + cirq.X.on_each(qubits_to_flip) + circuit[idx:]


def get_color(distance: int) -> str:
    if distance == 3:
        return "r"
    elif distance == 5:
        return "skyblue"
    elif distance == 7:
        return "b"


def get_marker(distance: int) -> str | tuple:
    if distance == 3:
        return "v"
    elif distance == 5:
        return "p"
    elif distance == 7:
        return (7, 0, 0)


class SurfaceCodeParams:
    """Contains parameters that describe a surface code experiment.

    Attributes:
        distance: The code distance.
        shift: Determines which qubits are used.
        observable: Which basis to measure in.
    """

    def __init__(
        self, distance: int, observable: Literal["H", "V"], shift: tuple[int, int]
    ):
        self.distance = distance
        self.shift = shift
        self.observable = observable


class LambdaExperimentResults:
    def __init__(
        self, cycles_list: list[int], repetitions: int, num_sweep_bit_choices: int
    ):
        self.lep_all = []
        self.cycles_list = cycles_list
        self.num_sweep_bit_choices = num_sweep_bit_choices
        self.repetitions = repetitions
        self.params_all = []
        self.shuffled_cycles_all = []
        self.avg_lep_by_distance = {}
        self.d_avg_lep_by_distance = {}
        self.fitted_ler = {}
        self.d_fitted_ler = {}
        self.fitted_scale = {}
        self.d_fitted_scale = {}

    def add_result(
        self, params: SurfaceCodeParams, lep: np.ndarray, shuffled_cycles: np.ndarray
    ):
        self.params_all.append(params)
        self.shuffled_cycles_all.append(shuffled_cycles)

        # unshuffle and reshape lep:
        order = np.argsort(shuffled_cycles)
        self.lep_all.append(
            lep[order].reshape(len(self.cycles_list), self.num_sweep_bit_choices)
        )

    def average_instances(self):
        distances = {params.distance for params in self.params_all}
        lep_by_distance = {distance: [] for distance in distances}
        for lep, params in zip(self.lep_all, self.params_all):
            lep_by_distance[params.distance].append(lep)
        for distance in distances:
            self.avg_lep_by_distance[distance] = np.mean(
                np.mean(lep_by_distance[distance], axis=0), axis=1
            )  # average over params and then sweep bits
            # the following is approximate and may underestimate the uncertainty
            # from sampling params if few params are sampled
            self.d_avg_lep_by_distance[distance] = np.std(
                np.array(lep_by_distance[distance])
                .transpose(0, 2, 1)
                .reshape(-1, len(self.cycles_list)),
                ddof=1,
                axis=0,
            ) / np.sqrt(self.num_sweep_bit_choices * len(lep_by_distance[distance]))

    def fit_exponential(self):
        self.average_instances()
        for distance, lep in self.avg_lep_by_distance.items():
            d_lep = self.d_avg_lep_by_distance[distance]
            if np.mean(d_lep) < 1e-5:
                d_lep += 1e-5  # to fit noiseless data
            popt, cov = fit_logical_error_per_cycle(self.cycles_list, lep, d_lep)
            self.fitted_scale[distance] = popt[0]
            self.d_fitted_scale[distance] = np.sqrt(cov[0, 0])
            self.fitted_ler[distance] = popt[1]
            self.d_fitted_ler[distance] = np.sqrt(cov[1, 1])

    def plot(self, ax: plt.Axes) -> plt.Axes:
        self.fit_exponential()
        included_distances = set()
        for params, lep in zip(self.params_all, self.lep_all):
            distance = params.distance
            marker = get_marker(distance)
            color = get_color(distance)
            for sweep_idx in range(self.num_sweep_bit_choices):
                ax.plot(
                    self.cycles_list,
                    lep[:, sweep_idx],
                    marker=marker,
                    color=color,
                    linestyle="none",
                    alpha=0.3,
                    label=(
                        f"$d = {params.distance}$ (individual)"
                        if distance not in included_distances
                        else None
                    ),
                )
                included_distances.add(distance)

        for distance, avg_lep in self.avg_lep_by_distance.items():
            marker = get_marker(distance)
            color = get_color(distance)
            d_avg_lep = self.d_avg_lep_by_distance[distance]
            ax.errorbar(
                self.cycles_list,
                avg_lep,
                d_avg_lep,
                marker=marker,
                color=color,
                linestyle="none",
                capsize=3,
                label=f"$d = {distance}$ (mean)",
                mec="k",
                ecolor="k",
                zorder=100,
            )
            t = np.linspace(0, 250, 100)
            ax.plot(
                t,
                (
                    1
                    - self.fitted_scale[distance]
                    * (1 - 2 * self.fitted_ler[distance]) ** t
                )
                / 2,
                color=color,
                label=f"Fit, LER={self.fitted_ler[distance]*100:.3f}% ± {self.d_fitted_ler[distance]*100:.3f}%",
            )

        if 3 in self.fitted_ler and 5 in self.fitted_ler:
            ax.text(
                175,
                0.5,
                f"$\\Lambda_{{35}} = {self.fitted_ler[3]/self.fitted_ler[5]:.2f} \\pm {np.sqrt( (self.d_fitted_ler[3]/self.fitted_ler[5])**2 + (self.fitted_ler[3]*self.d_fitted_ler[5]/self.fitted_ler[5]**2)**2 ):.2f}$",
                va="top",
            )
        if 5 in self.fitted_ler and 7 in self.fitted_ler:
            ax.text(
                175,
                0.5,
                f"\n$\\Lambda_{{57}} = {self.fitted_ler[5]/self.fitted_ler[7]:.2f} \\pm {np.sqrt( (self.d_fitted_ler[5]/self.fitted_ler[7])**2 + (self.fitted_ler[5]*self.d_fitted_ler[7]/self.fitted_ler[7]**2)**2 ):.2f}$",
                va="top",
            )

        ax.set_xlim(-1, 255)
        ax.set_ylim(-0.01, 0.55)
        ax.set_xlabel("Quantum error correction cycle, $t$")
        ax.set_ylabel("Logical error probability $p_L$")
        ax.legend(frameon=False, labelcolor="linecolor", loc="upper left")
        ax.tick_params(direction="in", top=True, right=True)

        return ax
