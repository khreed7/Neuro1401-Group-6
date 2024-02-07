import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.weights = np.random.random((height, width, input_dim))

    def train(self, inputs, num_epochs=100, initial_learning_rate=0.1, initial_radius=None):
        if initial_radius is None:
            initial_radius = max(self.height, self.width) / 2
        time_constant = num_epochs / np.log(initial_radius)
        
        for epoch in range(num_epochs):
            learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
            radius = initial_radius * np.exp(-epoch / time_constant)
            
            for input_vec in inputs:
                # Find the best matching unit
                bmu_idx = self.find_bmu(input_vec)
                
                # Update weights
                for i in range(self.height):
                    for j in range(self.width):
                        dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                        if dist_to_bmu <= radius:
                            influence = np.exp(-dist_to_bmu**2 / (2 * (radius**2)))
                            self.weights[i, j, :] += learning_rate * influence * (input_vec - self.weights[i, j, :])
    
    def find_bmu(self, input_vec):
        bmu_idx = np.argmin(np.linalg.norm(self.weights - input_vec, axis=-1))
        return np.unravel_index(bmu_idx, (self.height, self.width))
    
    def plot_som(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        for i in range(self.height):
            for j in range(self.width):
                ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1, facecolor=self.weights[i, j], edgecolor='none'))
        
        plt.show()

# Example usage
som = SOM(10, 10, 3)  # 10x10 grid with 3-dimensional input vectors (e.g., RGB colors)
inputs = np.random.random((100, 3))  # 100 random RGB colors
som.train(inputs, num_epochs=100)
som.plot_som()