import numpy as np
import matplotlib.pyplot as plt

class BoltzmannMachine:
    def __init__(self, n_units, learning_rate=0.1):
        self.n_units = n_units
        self.learning_rate = learning_rate
        self.weights = np.random.normal(0, 0.01, (n_units, n_units))
        np.fill_diagonal(self.weights, 0)
    
    def update_unit(self, unit_index, state):
        net_input = np.dot(self.weights[unit_index], state)
        probability = 1 / (1 + np.exp(-net_input))
        state[unit_index] = 1 if np.random.rand() < probability else 0
        return state
    
    def run_gibbs_sampling(self, initial_state, n_iterations=100):
        state = initial_state.copy()
        for _ in range(n_iterations):
            for i in range(self.n_units):
                state = self.update_unit(i, state)
        return state
    
    def train(self, data, epochs=10, n_gibbs_sampling_steps=100):
        wake_states = []  
        sleep_states = []  

        for _ in range(epochs):
            for data_point in data:
                data_state = np.array(data_point)
                wake_state = self.run_gibbs_sampling(data_state, n_gibbs_sampling_steps)
                
                positive_association = np.outer(data_state, data_state)
                negative_association_wake = np.outer(wake_state, wake_state)
                
                self.weights += self.learning_rate * (positive_association - negative_association_wake)
                
                wake_states.append(wake_state.copy())
            
            sleep_state_initial = np.random.choice([0, 1], self.n_units)
            sleep_state = self.run_gibbs_sampling(sleep_state_initial, n_gibbs_sampling_steps)
            negative_association_sleep = np.outer(sleep_state, sleep_state)
            
            activity = np.sum(negative_association_sleep, axis=0)
            less_active_connections = np.outer(activity, activity) < np.mean(activity) ** 2
            targeted_unlearning_rate = self.learning_rate * less_active_connections
            
            self.weights -= targeted_unlearning_rate * negative_association_sleep
            
            self.weights /= np.std(self.weights) 
            np.fill_diagonal(self.weights, 0)  
            
            sleep_states.append(sleep_state.copy())
        
        return wake_states, sleep_states
    
    def plot_memory_retrieval(self, states, phase_name):
        correlation_matrix = np.corrcoef(states, rowvar=False)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='viridis', origin='lower', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xlabel('Unit Index')
        plt.ylabel('Unit Index')
        plt.title(f'Correlation Matrix of Retrieved States After {phase_name} Phase')
        plt.show()

    def plot_initial_data_correlation(self, data):
        correlation_matrix = np.corrcoef(data, rowvar=False)
        
        # Plot the correlation matrix
        plt.figure(figsize=(8, 6))
        plt.imshow(correlation_matrix, cmap='viridis', origin='lower', vmin=-1, vmax=1)
        plt.colorbar(label='Correlation')
        plt.xlabel('Unit Index')
        plt.ylabel('Unit Index')
        plt.title('Correlation Matrix of Initial Data')
        plt.show()

n_units = 10
learning_rate = 0.1
data = np.random.choice([0, 1], (100, n_units))
data[:, :4] = np.random.choice([0, 1], (100, 1))

bm = BoltzmannMachine(n_units, learning_rate)
wake_states, sleep_states = bm.train(data, epochs=30, n_gibbs_sampling_steps=100)

bm.plot_memory_retrieval(wake_states, "Sleep")
bm.plot_memory_retrieval(sleep_states, "Wake")
bm.plot_initial_data_correlation(data)