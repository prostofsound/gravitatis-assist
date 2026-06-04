"""
------Simulate a gravity assist manouvrue for a probe flying by Mars, Jupiter and Saturn.-----

Author: Silva Alejandro
Date: March 2026
"""
##------ Import libraries ------##
import os
import sys
import subprocess

import configparser
import argparse

import scipy.constants as cte
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scienceplots
import pandas as pd

# Set plots style
plt.style.use(['science', 'notebook', 'no-latex'])

##------ Define the G constant -------##
G = cte.G # [m^3 kg^-1 s^-2]
AU = cte.au # [m]
SUN_MASS = 1.988416e30 # [kg]
YEAR = 365.25 * 24 * 3600 # [s]
# Getting G in AU
G_AU = G / (AU**3 / (SUN_MASS * YEAR**2)) # [AU^3 solar_mass^-1 year^-2]


##------ Define the main class ------##
class GravityAssistance:
    '''
    Initialize the 2D simulation using the slope of the ODE system.
    Also, it defines the different integrator methods.
    '''
    # Constructor
    def __init__(self, config_file, make_gif = False):
        '''
        Initialize the GravityAssistance simulation.

        Parameters
        ----------
        config_file : str
        Path to the configuration (.ini) file containing the simulation
        parameters.

        make_gif : bool, optional
        If True, an animation of the trajectory is generated and saved
        as a GIF after the simulation. The default is False.

        Notes
        -----
        The configuration file must contain a [simulation] section with
        the following parameters:

        x_s : float
            Initial x-shift of the probe (AU).
        b : float
            Impact parameter (AU).
        v0 : float
            Initial velocity of the probe (AU/year).
        M : float
            Mass of the planet (solar masses).
        planet : str
            Name of the planet.
        N : float
            Total integration time (years).
        method : str
            Numerical integration method ('rk2', 'rk4', or 'scipy').
        steps : int
            Number of integration steps for the simulation.

        Returns
        -------
        None
        '''
        self.gif_flag = make_gif
        self.load_config(config_file)

    # Read the config file
    def load_config(self, config_file):
        '''
        Load simulation parameters from a configuration (.ini) file.

        The configuration file must contain a [simulation] section. If any
        parameter is missing, a default value is used from the [DEFAULT]
        section defined in this method.

        Inputs
        ----------
        config_file : str
            Path to the configuration file.

        Returns
        ----------
        None
        '''
        # Initialize config parser
        config = configparser.ConfigParser()

        # Define default values
        config["DEFAULT"] = {
            "x_s": "0.3",
            "b": "0.01",
            "v0": "1.7",
            "M": "2.9e-4",
            "planet": "Saturn",
            "N": "0.4",
            "method": "rk4",
            "steps": "2000"
        }

        # Read config file
        config.read(config_file)

        # Read simulation section (will use DEFAULT if missing values)
        sim = config["simulation"]
        # Get the parameters for the simulation and set default parameters
        self.x_s = sim.getfloat("x_s")
        self.b = sim.getfloat("b")
        self.v0 = sim.getfloat("v0")
        self.M = sim.getfloat("M")
        self.planet = sim.get("planet")
        self.N = sim.getfloat("N")
        self.method = sim.get("method").lower()
        self.steps = sim.getint("steps")

        # Validate method
        if self.method not in ["rk2", "rk4", "scipy"]:
            raise ValueError("Method must be rk2, rk4 or scipy.")

    # Define initial conditions
    def initial_conditions(self):
        '''
        Initializes the two-body problem in the Cartesian plane.

        Inputs
        ----------
        x_s : float
            Initial horizontal offset (AU).
        b : float
            Impact parameter (AU).
        v0 : float
            Initial velocity magnitude (AU/year).

        Returns
        -------
        numpy.ndarray
        Initial state vector of the probe.
        '''
        # Get initial values of position and velocity
        x0 = -self.x_s
        y0 = self.b
        vx0 = self.v0
        vy0 = 0.0

        return np.array([x0, y0, vx0, vy0], dtype=float)

    # Slope of the ODE system
    def slope(self, _t, state):
        '''
        Computes the derivatives of the state vector for the gravitational
        two-body flyby problem.

        Inputs
        ----------
        t : float
            Time variable (years). Included for compatibility with ODE solvers,
            although the system is time-independent.
        state : numpy.ndarray
            State vector of the system.

        Returns
        -------
        numpy.ndarray
        Time derivative of the state vector.
        '''
        # Unpack the state variables
        x, y, vx, vy = state
        # Define the distance r
        r = np.sqrt(x**2 + y**2) + 1e-12 #to avoid singularities when r->0
        # Define the constant
        factor = -G_AU * self.M / r**3
        # Define the slope
        dx = vx
        dy = vy
        dvx = factor * x
        dvy = factor * y

        return np.array([dx, dy, dvx, dvy], dtype=float)

    # RK2 integrator
    def rk2(self):
        '''
        Solves the system of ODEs using the second-order Runge–Kutta
        method (Ralston scheme).

        Inputs
        ----------
        N : float
            Total integration time (years).
        steps : int
            Number of integration steps.
        initial_conditions : function
            Function that returns the initial state vector.
        slope : function
            Function defining the system of ODEs.

        Returns
        -------
        t : numpy.ndarray
            Time array from 0 to N (years).
        S : numpy.ndarray
            State array of shape (steps+1, 4), where each row contains:
            [x, y, vx, vy] at each time step.
        '''
        # Generate the time array
        t = np.linspace(0.0, self.N, self.steps + 1)
        # Define the step size
        h = t[1] - t[0]

        # Generate an empty state vector
        S = np.zeros((len(t), 4))
        # Add initial conditions
        S[0] = self.initial_conditions()
        # RK2 loop
        for j in range(len(t) - 1):
            k1 = self.slope(t[j], S[j])
            k2 = self.slope(t[j] + 3*h/4, S[j] + (3*h/4) * k1)
            S[j + 1] = S[j] + h * ((1/3)*k1 + (2/3)*k2)

        return t, S

    # RK4 method integrator
    def rk4(self):
        '''
        Solves the system of ODEs using the fourth-order Runge–Kutta
        method (3/8 scheme).
        Reference: https://lpsa.swarthmore.edu/NumInt/NumIntFourth.html

        Inputs
        ----------
        N : float
            Total integration time (years).
        steps : int
            Number of integration steps.
        initial_conditions : function
            Function that returns the initial state vector.
        slope : function
            Function defining the system of ODEs.

        Returns
        -------
        t : numpy.ndarray
            Time array from 0 to N (years).
        S : numpy.ndarray
            State array of shape (steps+1, 4), where each row contains:
            [x, y, vx, vy] at each time step.
        '''
        # Generate the time array
        t = np.linspace(0.0, self.N, self.steps + 1)
        # Define the step size
        h = t[1] - t[0]

        # Generate an empty state vector
        S = np.zeros((len(t), 4))
        # Add initial conditions
        S[0] = self.initial_conditions()
        # RK4 loop
        for j in range(len(t) - 1):
            k1 = self.slope(t[j], S[j])
            k2 = self.slope(t[j] + h/3, S[j] + (h/3) * k1)
            k3 = self.slope(t[j] + 2*h/3, S[j] - (h/3)*k1 + h*k2)
            k4 = self.slope(t[j] + h, S[j] + h*(k1 - k2 + k3))

            S[j + 1] = S[j] + (h/8)*(k1 + 3*k2 + 3*k3 + k4)

        return t, S

    # Scipy integrator
    def scipy_solver(self):
        '''
        Solves the ODE using SciPy's DOP853 integration method.

        Parameters
        ----------
        N : float
            Total integration time (years).
        steps : int
            Number of evaluation points.
        initial_conditions : function
            Function that returns the initial state vector.
        slope : function
            Function defining the system of ODEs.

        Returns
        -------
        t : numpy.ndarray
            Time array at which the solution was evaluated (years).
        y : numpy.ndarray
            State array of shape (steps+1, 4), where each row contains:
            [x, y, vx, vy] at each time step.
        '''
        # Generate the time array
        t = np.linspace(0.0, self.N, self.steps + 1)

        # Solving the ODE
        sol = solve_ivp(
            fun=self.slope,
            t_span=(0.0, self.N),
            y0=self.initial_conditions(),
            t_eval=t,
            method="DOP853",
            rtol=1e-12,
            atol=1e-12
        )

        return sol.t, sol.y.T #Transpose the solution to have each row as a time step


##------ Runner class ------##
class RunAssistance:
    '''
    Run and manage the gravity-assist simulation.
    '''
    # Constructor
    def __init__(self, model, output_dir="./outputfolder"):
        '''
        Initialize the RunAssistance class.

        Parameters
        ----------
        model : GravityAssistance
            Instance of the `GravityAssistance` class.
        output_dir : str
            Directory where output files will be stored. The default
            is "./outputfolder".

        Returns
        -------
        None
        '''
        self.model = model
        self.output_dir = output_dir
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

    # Integrator selector
    def integrator(self):
        '''
        Integrates the simulation using the numerical method selected in the model.
        Also, it saves the resulting time and state arrays inside a .csv file

        Returns
        -------
        t : numpy.ndarray
            Time array.
        S : numpy.ndarray
            State array of shape (steps+1, 4), where each row contains
            [x, y, vx, vy].
        '''
        # Get method and planet
        method = self.model.method.lower()
        planet = self.model.planet

        # Dictionary of available integrators
        methods = {
            "rk2": self.model.rk2,
            "rk4": self.model.rk4,
            "scipy": self.model.scipy_solver
        }

        if method not in methods:
            raise ValueError("Invalid method. Choose: rk2, rk4, scipy")

        # Run integrator
        t, S = methods[method]()

        # Save trajectory
        filename = f"traj_at_{planet}_{method}"
        self.save_csv(filename, t, S)

        # Message
        print(" ==> Integration finished and file saved successfully.\n")

        return t, S

    # Save trajectories
    def save_csv(self, filename, t, S):
        '''
        Saves the trajectory history into a CSV file using a Pandas DataFrame.

        Inputs
        ----------
        filename : str
            Name of the output CSV file, without extension.
        t : numpy.ndarray
            Time array.
        S : numpy.ndarray
            State array with columns [x, y, vx, vy].

        Returns
        -------
        None
        '''
        # Create the Pandas DataFrame using the columns of the state array and the time array
        df = pd.DataFrame({
            "t": t,
            "x": S[:, 0],
            "y": S[:, 1],
            "vx": S[:, 2],
            "vy": S[:, 3]
        })
        # Define the output path
        path = os.path.join(self.output_dir, f"{filename}.csv")
        # Save it as .csv file
        df.to_csv(path, index=False)


##------ Animation class ------##
class AnimateAssistance(RunAssistance):
    '''
    Handles trajectory loading and animation for the gravity-assist simulations.
    '''
    # Constructor inherited
    # Load trajectories
    def load_csv(self, filename):
        '''
        Loads a trajectory CSV file.

        Inputs
        ----------
        filename : str
            Path to the CSV file.

        Returns
        -------
        t : numpy.ndarray
            Time array.
        S : numpy.ndarray
            State array with columns [x, y, vx, vy].
        '''
        # Read the .csv file
        data = pd.read_csv(filename, sep=",")
        # Retrieve the columns
        time_arr = data["t"].to_numpy()
        state_arr = data[["x", "y", "vx", "vy"]].to_numpy()

        return time_arr, state_arr

    # Animation
    def animate_from_csv(self, csv_file, frame_step=8, window=0.35, interval=50):
        '''
        Generates a GIF animation from a trajectory CSV file using
        Matplotlib's FuncAnimation.

        The animation contains two panels:
            1. The trajectory of the probe near the planet.
            2. The speed magnitude as a function of time.

        Inputs
        ----------
        csv_file : str
            Path to the input CSV file containing the trajectory.
        frame_step : int
            Use every `frame_step`-th point from the trajectory in the
            animation. Larger values produce shorter GIFs.
            The default is 8.
        window : float
            Half-width of the square plotting window around the planet
            in AU. The default is 0.35.
        interval : int
            Delay between animation frames in milliseconds.
            The default is 60.

        Returns
        -------
        None
        '''
        # Get method and planet
        method = self.model.method.lower()
        planet = self.model.planet

        # Load CSV data
        t, S = self.load_csv(csv_file)

        # Unpack state variables
        x, y, vx, vy = S.T
        # Compute the speed
        vmag = np.sqrt(vx**2 + vy**2)

        # Get frame indices
        indices = np.arange(0, len(t), frame_step)
        if indices[-1] != len(t) - 1:
            indices = np.append(indices, len(t) - 1)

        # Fixed limits for the speed plot
        t_near = t[indices]
        v_near = vmag[indices]

        tmin, tmax = t_near[0], t_near[-1]
        vmin, vmax = np.min(v_near), np.max(v_near)
        dv = 0.05 * (vmax - vmin + 1e-12)

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6), constrained_layout=True)

        # -------- Left panel: trajectory --------
        # Plot the trajectory
        traj_line, = ax1.plot([], [], lw=1.2, alpha=0.9)
        # Fix the planet at the center
        ax1.scatter(0, 0, color='brown', s=500, label=planet)
        # Define the moving probe
        probe_point, = ax1.plot([], [], marker="*", color='darkblue', markersize=10,
                linestyle="None", label="Probe")

        # Fix the animation window centered on the planet
        ax1.set_xlim(-window, window)
        ax1.set_ylim(-window, window)
        # Ensure equal scale in x and y
        ax1.set_aspect("equal", adjustable="box")
        # Labels and title
        ax1.set_xlabel("x [AU]")
        ax1.set_ylabel("y [AU]")
        ax1.set_title(f"Trajectory of the probe near {planet}")
        ax1.grid(True, linestyle="--", linewidth=0.8)
        ax1.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1)

        # -------- Right panel: speed magnitude --------
        # Plot the velocity trajectory
        speed_line, = ax2.plot([], [], lw=1.2, linestyle="--", color='black',
                label=r"$|\vec{v}|$")
        # Define the moving point
        speed_point, = ax2.plot([], [], marker="*", color='darkblue',
                markersize=10, linestyle="None")

        # Fix limits for a stable animation
        ax2.set_xlim(tmin, tmax)
        ax2.set_ylim(vmin - dv, vmax + dv)
        # Lables and title
        ax2.set_xlabel("t [yr]")
        ax2.set_ylabel(r"$|\vec{v}|$ [AU/yr]")
        ax2.set_title("Velocity magnitude vs time")
        ax2.grid(True, linestyle="--", linewidth=0.8)
        ax2.legend(frameon=True, facecolor="white", edgecolor="black", framealpha=1)

        # Begin the animation
        def init():
            '''
            Initialize empty animation artists.
            '''
            # Set data for every animation
            traj_line.set_data([], [])
            probe_point.set_data([], [])
            speed_line.set_data([], [])
            speed_point.set_data([], [])
            return traj_line, probe_point, speed_line, speed_point

        # Upadate frames
        def update(frame):
            '''
            Update animation for a given frame.
            This replaces a for loop we would have done if every frame was first saved
            and then animated to a gif.
            '''
            # Define an index for each frame
            i = indices[frame]

            # Left panel
            traj_line.set_data(x[indices[:frame+1]], y[indices[:frame+1]])
            probe_point.set_data([x[i]], [y[i]])
            # Set title with timestamp
            ax1.set_title(f"Trajectory at t = {t[i]:.4f} yr")

            # Right panel
            speed_line.set_data(t[indices[:frame+1]], vmag[indices[:frame+1]])
            speed_point.set_data([t[i]], [vmag[i]])

            return traj_line, probe_point, speed_line, speed_point

        # Main animation
        anim = FuncAnimation(
            fig,
            update,
            frames=len(indices),
            init_func=init,
            interval=interval,
            blit=False
        )

        # Build output path
        gif_path = os.path.join(self.output_dir, f"traj_at_{planet}_{method}.gif")

        # Save GIF
        anim.save(gif_path, writer="pillow", fps=max(1, 1000 // interval))

        plt.close(fig)
        print(" ==> The GIF has been successfully generated.")
        print(f" ==> GIF saved to: {gif_path}")

    def open_gif(self, gif_path):
        '''
        Opens the GIF file using the system default viewer.
        Works on Linux, macOS, and Windows.
        '''
        try:
            # Linux
            if sys.platform.startswith("linux"):
                subprocess.run(["xdg-open", gif_path], check=True)
            # Mac
            elif sys.platform == "darwin":
                subprocess.run(["open", gif_path], check=True)
            # Windows
            elif sys.platform == "win32":
                os.startfile(gif_path)
            else:
                print("Unsupported OS. Cannot open GIF automatically.")
        except Exception as error:
            print(f"Could not open GIF automatically:{error}.")


# Parser
def parse_args():
    """
    Set up the command line arguments for both the
    integrator and the animation classes.

    Returns
    -------
    parser.parse_args()
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="Gravity assist simulation")

    parser.add_argument("--config", type=str, required=True,
                        help="Path to config.ini file.")

    parser.add_argument("--gif", action="store_true",
                        help="Generate and open GIF animation.")

    return parser.parse_args()

# ----------------- Main ----------------

if __name__ == "__main__":

    # Initialize the parser
    args = parse_args()

    # Create simulation model
    sim = GravityAssistance(args.config)


    # Package initialization message
    print("")
    print("╔══════════════════════════════════════════════════════╗")
    print("║            GRAVITY ASSISTANCE SIMULATION             ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║ Planet:                {sim.planet:<30}║")
    print(f"║ Method:                {sim.method.upper():<30}║")
    print(f"║ Simulation time (yr):  {sim.N:<30}║")
    print(f"║ Steps:                 {sim.steps:<30}║")
    print("╚══════════════════════════════════════════════════════╝")
    print("")

    print("Running simulation...")

    # Run simulation
    runner = RunAssistance(sim)
    time_array, S_array = runner.integrator()

    # If GIF requested
    if args.gif:
        # Message
        print("Generating GIF...")
        animator = AnimateAssistance(sim)
        CSV_FILE = os.path.join(
                animator.output_dir,
                f"traj_at_{sim.planet}_{sim.method.lower()}.csv"
                )

        animator.animate_from_csv(CSV_FILE)

        GIF_FILE = os.path.join(
                animator.output_dir,
                f"traj_at_{sim.planet}_{sim.method.lower()}.gif"
                )
        animator.open_gif(GIF_FILE)
