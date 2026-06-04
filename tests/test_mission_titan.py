# Import pytest
import numpy as np
import scipy.constants as cte
import pytest

# Import our module
from assisted_probe import GravityAssistance, RunAssistance

##------ Define the G constant -------##
G = cte.G # [m^3 kg^-1 s^-2]
AU = cte.au # [m]
SUN_MASS = 1.988416e30 # [kg]
YEAR = 365.25 * 24 * 3600 # [s]
# Getting G in AU
G_AU = G / (AU**3 / (SUN_MASS * YEAR**2)) # [AU^3 solar_mass^-1 year^-2]


# Write a config file with the default parameters
def write_config(tmp_path, **kwargs):
    '''
    Creates a temporary config.ini file for testing.

    Inputs
    ----------
    tmp_path : pathlib.Path
        A pytest fixture that provides a temporary directory.
        Pytest automatically creates this directory and deletes it after the test.

    **kwargs : dict
        Optional parameters that override the default simulation values.

    Returns
    -------
    config_file : pathlib.Path
        Path to the temporary config.ini file that was created.
    '''
    # Define the default parameters dictionary
    defaults = {
        "x_s": 0.3,
        "b": 0.01,
        "v0": 1.7,
        "M": 2.9e-4,
        "planet": "Saturn",
        "N": 0.4,
        "method": "rk4",
        "steps": 2000,
    }
    # Update default parameters if a custom parameter is defined
    defaults.update(kwargs)
    # Create the text content of the .ini file
    config_text = f"""
[simulation]
x_s = {defaults["x_s"]}
b = {defaults["b"]}
v0 = {defaults["v0"]}
M = {defaults["M"]}
planet = {defaults["planet"]}
N = {defaults["N"]}
method = {defaults["method"]}
steps = {defaults["steps"]}
"""
    # Create the file path inside the temporary pytest directory
    config_file = tmp_path / "config.ini"

    # Write the config text into that file
    config_file.write_text(config_text.strip())

    # Return the path of the config file so the test can use it
    return config_file


#Define a fixture, it's a reusable object that tests can use as input.
@pytest.fixture
def simulation(tmp_path):
    '''
    Pytest fixture that creates a GravityAssistance simulation object
    using a temporary config.ini file.

    Inputs
    ----------
    tmp_path : pathlib.Path
        A pytest fixture that provides a temporary directory.
        Pytest automatically creates this directory and deletes it after the test.

    Returns
    -------
    GravityAssistance
        An initialized simulation object using the default parameters.
    '''
    # Create a temporary config.ini file inside the pytest temporary folder
    config_file = write_config(tmp_path)

    # Return a GravityAssistance object recalling that it receives a file path
    return GravityAssistance(str(config_file))

# Testing class
class TestGravityAssistance:
    '''
    Tests the input mapping, confirms supported methods and verifies energy conservation
    within some tolerance.
    '''
    # Testing input mapping
    def test_input_mapping(self, tmp_path):
        '''
        Verify that the user-defined input parameters are correctly mapped
        to the initial state vector.
        This test checks that:
            x0  = -x_s
            y0  = b
            vx0 = v0
            vy0 = 0
        '''
        # Create a temporary config file with custom parameters
        config_file = write_config(tmp_path, x_s=0.5, b=0.07, v0=1.8)

        # Initialize the simulation using this custom config file
        simulation = GravityAssistance(str(config_file))

        # Expected initial state vector
        expected = np.array([-0.5, 0.07, 1.8, 0.0])

        # Check match between initial state vector computed with our module
        # and the expected initial state vector
        # numpy.allclose() checks if two arrays are element-wise equal, we
        # are comparing float numbers.
        assert np.allclose(simulation.initial_conditions(), expected)

    # Testing supported methods
    def test_invalid_method(self, tmp_path):
        '''
        Verifies that the module correctly handles unsupported integration methods.
        The test is passed if the ValueError is raised, and fails otherwise.
        '''
        # Create a temporary config file with an unsupported method
        config_file = write_config(tmp_path, method="RK3")

        # The GravityAssistance constructor should validate the method
        with pytest.raises(ValueError):
            GravityAssistance(str(config_file)) # Instance class with invalid method

    # Testing energy conservation
    def test_energy_cons(self, tmp_path):
        '''
        Verify that the specific orbital energy remains approximately constant
        during the simulation.

        In a two-body gravitational system with a fixed central mass, the
        specific orbital energy is conserved:

            Energy = v^2 / 2 - GM / r = cte.

        This test runs a simulation using the RK4 method.
        Since RK4 is a fourth-order method, the energy error is expected to be small,
        so a tolerance of e-4 is used.
        '''
        # Create a temporary config file with a large number of integration steps.
        config_file = write_config(tmp_path, method="rk4", steps=4000)
        # Initialize the simulation using this custom config file
        simulation = GravityAssistance(str(config_file))

        # Perform the numerical integration
        _, state = RunAssistance(simulation, output_dir=tmp_path).integrator()

        # Extract position and velocity components
        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        vy = state[:, 3]

        # Compute the radial distance from the planet at each time step
        r = np.sqrt(x**2 + y**2)

        # Compute the velocity term of the energy
        v2 = vx**2 + vy**2
        # Compute the energy
        energy = 0.5 * v2 - G_AU * simulation.M / r

        # Compare computed energy at all time steps
        # to the initial energy value within some tolerance
        assert np.allclose(energy, energy[0], rtol=1e-4, atol=1e-6)
