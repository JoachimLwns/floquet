import numpy as np
from pylab import norm
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from typing import Callable, Dict, List, Tuple, Optional

class FloquetSystem:
    '''
    Class for calculating Floquet modes, quasi-energies and time evolution of driven two-level systems.
    '''

    def __init__(self, frequency_cutoff: int):
        """
       Initialize the Floquet system.

       Parameters
       ----------
       frequency_cutoff : int
           Determines the size of the Floquet Hamiltonian matrix (precision).
       """
        self.frequency_cutoff = frequency_cutoff
        self.matrix_dimension = (4 * frequency_cutoff) + 2
        self.hbar = 1.0
        self.floquet_Hamiltonian = np.zeros((self.matrix_dimension, self.matrix_dimension), dtype=complex)
        self.static_Hamiltonian = np.zeros((2, 2), dtype=complex)

        # Floquet operators
        self.static_sigma_z = np.diag([((-1) ** (n + 0)) for n in range(self.matrix_dimension)])
        self.static_sigma_x = np.diag([1 if n % 2 == 0 else 0 for n in range(self.matrix_dimension - 1)], 1) + np.diag([1 if n % 2 == 0 else 0 for n in range(self.matrix_dimension - 1)], -1)
        self.static_sigma_y = np.diag([1j if n % 2 == 0 else 0 for n in range(self.matrix_dimension - 1)], 1) + np.diag([-1j if n % 2 == 0 else 0 for n in range(self.matrix_dimension - 1)], -1)
        self.static_operator_names = {
            'sigma_z': self.static_sigma_z,
            'sigma_x': self.static_sigma_x,
            'sigma_y': self.static_sigma_y
        }

        # 2x2 projection operators
        self.projection_operators = {
            'sigma_z': np.array([[1, 0], [0, -1]]),
            'sigma_x': np.array([[0, 1],[1, 0]]),
            'sigma_y': np.array([[0, -1j],[1j, 0]])
        }

    def generate_Hamiltonian(
            self,
            static_Hamiltonian_terms: Dict[str, float],
            drive_Hamiltonian_terms: Dict[str, List[List[float]]],
            omega: float
    ) -> Tuple[np.ndarray, Callable[[float], np.ndarray]]:
        """
        Generates the Fourier-Floquet Hamiltonian and a time-dependent Hamiltonian function.

        Parameters
        ----------
        static_Hamiltonian_terms : dict
            Dictionary of static Hamiltonian terms {'operator_name': amplitude}.
        drive_Hamiltonian_terms : dict
            Dictionary of drive terms {'operator_name': [[amplitude, frequency_multiplier], ...]}.
        omega : float
            Driving frequency.

        Returns
        -------
        Tuple[np.ndarray, Callable[[float], np.ndarray]]
            Floquet Hamiltonian and a function H(t) returning the instantaneous Hamiltonian.
        """

        self.omega = omega

        for operator, amplitude in static_Hamiltonian_terms.items():
            self.floquet_Hamiltonian += amplitude * self.static_operator_names[operator]
            self.static_Hamiltonian += amplitude * self.projection_operators[operator]

        drive_lambdas = []

        for operator, drive_parameters in drive_Hamiltonian_terms.items():
            for [amplitude, frequency_multiplier] in drive_parameters:

                if operator == 'drive_sigma_z':
                    operator_matrix = np.diag([((-1) ** (i + 0)) * amplitude / 2 for i in range(self.matrix_dimension - 2 * frequency_multiplier)], 2 * frequency_multiplier) + np.diag([((-1) ** (i + 0)) * amplitude / 2 for i in range(self.matrix_dimension - 2 * frequency_multiplier)], -2 * frequency_multiplier)
                    self.floquet_Hamiltonian += operator_matrix

                    drive_lambdas.append(lambda t, a=amplitude, fm=frequency_multiplier: a * np.cos(fm * omega * t) * self.projection_operators['sigma_z'])

                elif operator == 'drive_sigma_x':
                    operator_matrix_1 = np.diag([amplitude / 2 if i%2!=0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier + 1)], 2 * frequency_multiplier - 1) + np.diag([amplitude / 2 if i%2==0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier - 1)], 2 * frequency_multiplier + 1)
                    operator_matrix_2 = np.diag([amplitude / 2 if i%2!=0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier + 1)], -2 * frequency_multiplier + 1) + np.diag([amplitude / 2 if i%2==0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier - 1)], -2 * frequency_multiplier - 1)
                    self.floquet_Hamiltonian += operator_matrix_1 + operator_matrix_2

                    drive_lambdas.append(lambda t, a=amplitude, fm=frequency_multiplier: a * np.cos(fm * omega * t) * self.projection_operators['sigma_x'])


                elif operator == 'drive_sigma_y':
                    operator_matrix_1 = np.diag([1j * amplitude / 2 if i%2!=0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier + 1)], 2 * frequency_multiplier - 1) + np.diag([-1j * amplitude / 2 if i%2==0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier - 1)], 2 * frequency_multiplier + 1)
                    operator_matrix_2 = np.diag([-1j * amplitude / 2 if i%2!=0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier + 1)], -2 * frequency_multiplier + 1) + np.diag([1j * amplitude / 2 if i%2==0 else 0 for i in range(self.matrix_dimension - 2 * frequency_multiplier - 1)], -2 * frequency_multiplier - 1)
                    self.floquet_Hamiltonian += operator_matrix_1 + operator_matrix_2

                    drive_lambdas.append(lambda t, a=amplitude, fm=frequency_multiplier: a * np.cos(fm * omega * t) * self.projection_operators['sigma_y'])

        time_derivative = lambda x: x * self.omega * self.hbar
        time_derivative_part = np.diag([func(n) for n in range(-self.frequency_cutoff, self.frequency_cutoff + 1) for func in (time_derivative, time_derivative)])
        self.floquet_Hamiltonian += time_derivative_part

        drive_terms = lambda t: sum(f(t) for f in drive_lambdas)

        self.time_dependent_Hamiltonian = lambda t: self.static_Hamiltonian + drive_terms(t)

        return self.floquet_Hamiltonian, self.time_dependent_Hamiltonian

    def diagonalize(self, Hamiltonian: Optional[np.ndarray] = None) -> tuple:
        """
        Diagonalize the Hamiltonian to obtain quasi-energies and Floquet modes.

        Parameters
        ----------
        Hamiltonian : np.ndarray, optional
            Hamiltonian matrix to diagonalize (defaults to self.floquet_Hamiltonian).

        Returns
        -------
        List[Tuple[float, np.ndarray]]
            List of tuples (eigenvalue, eigenvector) for the two lowest quasi-energies.
        """
        H = Hamiltonian if Hamiltonian is not None else self.floquet_Hamiltonian

        sparse_Hamiltonian = csr_matrix(H)
        eigenvalues, eigenvectors = eigsh(sparse_Hamiltonian, sigma=0, k=4)
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        idx = self.get_smallest_eigenvalues(eigenvalues)  # Only use the two eigenvalues closest to zero, the other ones are exact copies shifted by nhw.
        indep_values = [eigenvalues[idx], eigenvalues[idx + 1]]
        indep_vectors = [eigenvectors[:, idx], eigenvectors[:, idx + 1]]
        vector_0 = indep_vectors[0]
        vector_1 = indep_vectors[1]

        upvector_0 = vector_0[::2]
        downvector_0 = vector_0[1::2]
        normalization_factor_0 = norm([sum(upvector_0), sum(downvector_0)])  # normalize the Floquet modes

        upvector_1 = vector_1[::2]
        downvector_1 = vector_1[1::2]
        normalization_factor_1 = norm([sum(upvector_1), sum(downvector_1)])  # normalize the Floquet modes

        self.upvector_0 = [element / normalization_factor_0 for element in upvector_0]
        self.downvector_0 = [element / normalization_factor_0 for element in downvector_0]

        self.upvector_1 = [element / normalization_factor_1 for element in upvector_1]
        self.downvector_1 = [element / normalization_factor_1 for element in downvector_1]

        self.energy_0 = indep_values[0]
        self.energy_1 = indep_values[1]

        return [indep_values[0], vector_0], [indep_values[1], vector_1]

    def get_smallest_eigenvalues(self, values: np.ndarray) -> int:
        """
        Identify index of eigenvalues closest to zero.

        Parameters
        ----------
        values : np.ndarray
            Sorted eigenvalues.

        Returns
        -------
        int
            Index of eigenvalue just below zero.
        """
        for i in range(len(values) - 1):
            if values[i] < 0 and values[i + 1] > 0:
                return i
        raise Exception('Eigenvalues around zero not found :(')

    def plot_floquet_modes(self, amount_of_timesteps: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Plot the time evolution of the Floquet modes on the Bloch sphere over one drive period.

        Parameters
        ----------
        amount_of_timesteps : int, optional
            Number of time steps to compute the trajectory (default=200).

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Arrays containing Bloch sphere coordinates [x, y, z] for Floquet mode 0 and 1.
        """

        # --- Prepare ---
        vector_length = len(self.upvector_0)
        frequency_vector = np.array([1j * k * self.omega for k in range((-vector_length // 2) + 1, (vector_length // 2) + 1)])
        timesteps = np.linspace(0, 2 * np.pi / self.omega, amount_of_timesteps)

        # --- Compute Bloch coordinates at each timestep ---
        bloch_x_0, bloch_y_0, bloch_z_0 = [], [], []
        bloch_x_1, bloch_y_1, bloch_z_1 = [], [], []

        for current_time in timesteps:
            up_amplitude_0 = sum(self.upvector_0[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            down_amplitude_0 = sum(self.downvector_0[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))

            # Normalize spinor
            psi_0 = np.array([up_amplitude_0, down_amplitude_0], dtype=complex)
            psi_0 /= np.linalg.norm(psi_0)

            # Compute Bloch coordinates
            c_up_0, c_down_0 = psi_0
            sx_0 = 2 * np.real(np.conj(c_up_0) * c_down_0)
            sy_0 = 2 * np.imag(np.conj(c_up_0) * c_down_0)
            sz_0 = np.abs(c_up_0) ** 2 - np.abs(c_down_0) ** 2

            bloch_x_0.append(sx_0)
            bloch_y_0.append(sy_0)
            bloch_z_0.append(sz_0)

            # Second Floquet mode
            up_amplitude_1 = sum(self.upvector_1[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            down_amplitude_1 = sum(self.downvector_1[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))

            # Normalize spinor
            psi_1 = np.array([up_amplitude_1, down_amplitude_1], dtype=complex)
            psi_1 /= np.linalg.norm(psi_1)

            # Compute Bloch coordinates
            c_up_1, c_down_1 = psi_1
            sx_1 = 2 * np.real(np.conj(c_up_1) * c_down_1)
            sy_1 = 2 * np.imag(np.conj(c_up_1) * c_down_1)
            sz_1 = np.abs(c_up_1) ** 2 - np.abs(c_down_1) ** 2

            bloch_x_1.append(sx_1)
            bloch_y_1.append(sy_1)
            bloch_z_1.append(sz_1)


        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Draw sphere surface
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax.plot_surface(x, y, z, color='lightgrey', alpha=0.1, edgecolor='none')

        # Equator and axes
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 0, 'k', lw=0.5, alpha=0.3)
        ax.plot([-1, 1], [0, 0], [0, 0], 'k', alpha=0.2)
        ax.plot([0, 0], [-1, 1], [0, 0], 'k', alpha=0.2)
        ax.plot([0, 0], [0, 0], [-1, 1], 'k', alpha=0.2)

        # Trajectory of the Bloch vector
        ax.plot(bloch_x_0, bloch_y_0, bloch_z_0, color='blue', lw=1.5, label='Floquet mode 0')
        ax.plot(bloch_x_1, bloch_y_1, bloch_z_1, color='red', lw=1.5, label='Floquet mode 1')


        # Plot settings
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_title("Floquet mode")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return np.array([bloch_x_0, bloch_y_0, bloch_z_0]), np.array([bloch_x_1, bloch_y_1, bloch_z_1])

    def sweep_values(
            self,
            operator: str,
            values: np.ndarray = np.linspace(-2, 2, 500),
            make_plot: bool = True,
            plot_copies: bool = True
    ) -> Tuple[List[float], List[float]]:
        """
        Sweep a parameter and compute the corresponding quasi-energies.

        Parameters
        ----------
        operator : str
            Operator to sweep ('sigma_x', 'sigma_y', or 'sigma_z').
        values : np.ndarray, optional
            Values to sweep (default=np.linspace(-2, 2, 500)).
        make_plot : bool, optional
            Whether to generate a plot (default=True).
        plot_copies : bool, optional
            Whether to plot shifted copies by ±ω (default=True).

        Returns
        -------
        Tuple[List[float], List[float]]
            Quasi-energy arrays for Floquet mode 0 and 1.
        """

        length = len(values)
        sweep_energy_0 = length * [0, ]
        sweep_energy_1 = length * [0, ]
        sweep_energy_0_copy = length * [0, ]
        sweep_energy_1_copy = length * [0, ]

        for idx, value in enumerate(values):
            H_sweep = self.floquet_Hamiltonian.copy() + value * self.static_operator_names[operator]
            [E_0, V_0], [E_1, V_1] = self.diagonalize(H_sweep)
            sweep_energy_0[idx] = E_0
            sweep_energy_1[idx] = E_1
            sweep_energy_0_copy[idx] = E_0 + self.omega
            sweep_energy_1_copy[idx] = E_1 - self.omega

        if make_plot:
            plt.plot(values, sweep_energy_0, color='k')
            plt.plot(values, sweep_energy_1, color='k')
            if plot_copies:
                plt.plot(values, sweep_energy_0_copy, color='grey', linestyle='dashed')
                plt.plot(values, sweep_energy_1_copy, color='grey', linestyle='dashed')
            plt.show()

        return sweep_energy_0, sweep_energy_1

    def time_evolution_rk4(
            self,
            psi_0: np.ndarray,
            t_span: Tuple[float, float] = (0.0, 50.0),
            dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the time-dependent Schrödinger equation using RK4.

        Parameters
        ----------
        psi_0 : np.ndarray
            Initial 2-level state vector.
        t_span : tuple
            (start_time, end_time)
        dt : float
            Time step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time points and array of states at each time step (shape=(N,2)).
        """
        t0, tf = t_span
        t_points = np.arange(t0, tf, dt)
        psi_t = np.zeros((len(t_points), len(psi_0)), dtype=complex)

        psi = np.array(psi_0, dtype=complex)
        psi_t[0] = psi

        def schrodinger_rhs(t, psi):
            """Computes dψ/dt = -i / ħ * H(t) ψ"""
            return -1j / self.hbar * self.time_dependent_Hamiltonian(t) @ psi

        for i in range(1, len(t_points)):
            t = t_points[i - 1]
            k1 = schrodinger_rhs(t, psi)
            k2 = schrodinger_rhs(t + dt / 2, psi + dt * k1 / 2)
            k3 = schrodinger_rhs(t + dt / 2, psi + dt * k2 / 2)
            k4 = schrodinger_rhs(t + dt, psi + dt * k3)
            psi = psi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            psi /= np.linalg.norm(psi)
            psi_t[i] = psi

        return t_points, psi_t

    def time_evolution_floquet(
            self,
            psi_0: np.ndarray,
            t_span: Tuple[float, float] = (0.0, 50.0),
            dt: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute time evolution of an initial state using Floquet modes.

        Parameters
        ----------
        psi_0 : np.ndarray
            Initial 2-level state vector.
        t_span : tuple
            (start_time, end_time)
        dt : float
            Time step.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Time points and state evolution array (shape=(N,2)).
        """

        # --- Prepare time grid ---
        timesteps = np.arange(t_span[0], t_span[1], dt)
        vector_length = len(self.upvector_0)
        frequency_vector = np.array([1j * k * self.omega for k in range((-vector_length // 2) + 1, (vector_length // 2) + 1)])

        # --- Compute Floquet modes at t=0 ---
        up0_t0 = np.array([sum(self.upvector_0[i] for i in range(vector_length))])
        down0_t0 = np.array([sum(self.downvector_0[i] for i in range(vector_length))])
        floquet_mode_0_t0 = np.array([up0_t0, down0_t0], dtype=complex).flatten()

        up1_t0 = np.array([sum(self.upvector_1[i] for i in range(vector_length))])
        down1_t0 = np.array([sum(self.downvector_1[i] for i in range(vector_length))])
        floquet_mode_1_t0 = np.array([up1_t0, down1_t0], dtype=complex).flatten()

        # --- Project initial state onto Floquet basis ---
        a = np.vdot(floquet_mode_0_t0, psi_0)
        b = np.vdot(floquet_mode_1_t0, psi_0)

        # --- Initialize storage ---
        psi_t = np.zeros((len(timesteps), 2), dtype=complex)

        # --- Time evolution ---
        for idx, current_time in enumerate(timesteps):
            # Compute instantaneous Floquet modes
            up0 = sum(self.upvector_0[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            down0 = sum(self.downvector_0[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            mode0 = np.array([up0, down0])

            up1 = sum(self.upvector_1[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            down1 = sum(self.downvector_1[i] * np.exp(frequency_vector[i] * current_time) for i in range(vector_length))
            mode1 = np.array([up1, down1])

            # Time evolution formula
            psi = (
                    a * mode0 * np.exp(-1j * self.energy_0 * current_time)
                    + b * mode1 * np.exp(-1j * self.energy_1 * current_time)
            )

            # Normalize and store
            psi /= np.linalg.norm(psi)
            psi_t[idx] = psi

        return timesteps, psi_t

    def stroboscopic_evolution(
            self,
            psi_0: np.ndarray,
            n_periods: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute stroboscopic evolution at integer multiples of the driving period.

        Parameters
        ----------
        psi_0 : np.ndarray
            Initial 2-level state vector.
        n_periods : int
            Number of driving periods to simulate.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Array of stroboscopic times and state evolution (shape=(n_periods+1,2)).
        """
        T = 2 * np.pi / self.omega
        timesteps = np.arange(0, n_periods + 1) * T

        vector_length = len(self.upvector_0)

        # --- Floquet modes at t=0 ---
        up0_t0 = sum(self.upvector_0[i] for i in range(vector_length))
        down0_t0 = sum(self.downvector_0[i] for i in range(vector_length))
        floquet_mode_0_t0 = np.array([up0_t0, down0_t0], dtype=complex)

        up1_t0 = sum(self.upvector_1[i] for i in range(vector_length))
        down1_t0 = sum(self.downvector_1[i] for i in range(vector_length))
        floquet_mode_1_t0 = np.array([up1_t0, down1_t0], dtype=complex)

        # --- Project initial state onto Floquet basis ---
        a = np.vdot(floquet_mode_0_t0, psi_0)
        b = np.vdot(floquet_mode_1_t0, psi_0)

        # --- Time evolution (stroboscopic only) ---
        psi_t = np.zeros((len(timesteps), 2), dtype=complex)
        for idx, t in enumerate(timesteps):
            psi = (
                a * floquet_mode_0_t0 * np.exp(-1j * self.energy_0 * t)
                + b * floquet_mode_1_t0 * np.exp(-1j * self.energy_1 * t)
            )
            psi /= np.linalg.norm(psi)
            psi_t[idx] = psi

        return timesteps, psi_t
