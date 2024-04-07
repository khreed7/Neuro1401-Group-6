import numpy as np
from scipy.stats import poisson
from scipy.special import i0
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

M = 100  # number of neurons in the subpopulation
stimulus_orientation = 45  # orientation of the stimulus in degrees
neuron_preferences = np.linspace(0, 180, M)  # preferred orientations of the neurons
omega = 5  # concentration parameter
response_noise_std = 5  # standard deviation of response noise
gamma = 0.1  # normalization constant
alpha_p = 1  # attentional gain factor for the true stimulus location
alpha_n = np.ones(M)  # attentional gain factors for the neurons
T_d = 1  # decoding time window in seconds

# tuning curve function
def tuning_curve(stimulus_orientation, neuron_preference, omega):
    return np.exp(omega * (np.cos(np.deg2rad(neuron_preference - stimulus_orientation)) - 1))

# divisive normalization function
def divisive_normalization(responses, alpha, gamma):
    summed_activity = np.sum(alpha * responses**2)
    normalized_responses = gamma * (alpha * responses) / summed_activity
    return normalized_responses

# simulate neuron responses to the stimulus orientation
responses = tuning_curve(stimulus_orientation, neuron_preferences, omega) + np.random.normal(0, response_noise_std, M)
normalized_responses = divisive_normalization(responses, alpha_n, gamma)

# probability density function 
def probability_density_epsilon(epsilon, omega):
    return np.exp(omega**(-1) * np.cos(epsilon)) / (2 * np.pi * i0(omega**(-1)))

def cos_sum(theta, epsilon):
    return -np.sum(np.cos(np.deg2rad(theta) - epsilon))  

# generate a sample of epsilon values based on the preferred orientations and true stimulus orientation
epsilon_sample = np.deg2rad(neuron_preferences - stimulus_orientation)

# calculate the expected value xi for the total spike count m
xi = gamma * T_d * alpha_p / np.sum(alpha_n)

# find decoding error
result = minimize_scalar(cos_sum, args=(epsilon_sample,), bounds=(0, 2*np.pi), method='bounded')
delta_theta = result.x  

# simulation parameteers
num_simulations = 1000 
error_range = np.linspace(-np.pi, np.pi, 100)  
num_stimuli_range = [1, 2, 4, 8]

# simulate decoding error
def simulate_decoding_error(num_stimuli, num_simulations, omega, gamma, alpha_n, T_d, neuron_preferences):
    decoding_errors = []
    for _ in range(num_simulations):
        selected_preferences = np.random.choice(neuron_preferences, num_stimuli, replace=False)
        epsilon_sample = np.deg2rad(selected_preferences - stimulus_orientation)
        result = minimize_scalar(cos_sum, args=(epsilon_sample,), bounds=(0, 2*np.pi), method='bounded')
        decoding_error = result.x
        decoding_errors.append(decoding_error)
    return np.array(decoding_errors)

# plot distribution of recall errors 
plt.figure(figsize=(12, 8))

num_simulations = 1000  
for num_stimuli in num_stimuli_range:
    errors = simulate_decoding_error(num_stimuli, num_simulations, omega, gamma, alpha_n, T_d, neuron_preferences)
  
    normalized_errors = (errors + np.pi) % (2 * np.pi) - np.pi
    
    # plot probability density
    density, bins = np.histogram(normalized_errors, bins=50, range=(-np.pi, np.pi), density=True)
    centers = (bins[:-1] + bins[1:]) / 2
    plt.plot(centers, density, label='{} Stimuli'.format(num_stimuli))

plt.xlabel('Error (Radians)')
plt.ylabel('Probability Density')
plt.title('Distribution of Recall Errors for Different Numbers of Orientation Stimuli')
plt.legend()
plt.xlim(-np.pi, np.pi)
plt.show()