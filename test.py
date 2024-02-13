import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class SOM:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.weights = np.random.random((height, width, input_dim))

    def train(self, inputs, num_epochs=100, initial_learning_rate=0.5, initial_radius=None):
        #NOTE: article uses initial_learning_rate = 0.01, but 0.1 produces better results for us for now
        if initial_radius is None:
            initial_radius = max(self.height, self.width) / 2
        time_constant = num_epochs / np.log(initial_radius)

        #adjusting learning rate and weights
        for epoch in range(num_epochs):
            learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
            radius = initial_radius * np.exp(-epoch / time_constant)

            for input_vec in inputs:
                bmu_idx = self.find_bmu(input_vec)

                for i in range(self.height):
                    for j in range(self.width):
                        dist_to_bmu = np.linalg.norm(np.array([i, j]) - np.array(bmu_idx))
                        if dist_to_bmu <= radius:
                            influence = np.exp(-dist_to_bmu**2 / (2 * (radius**2)))
                            #implementing hebbian to update weights at each time-step
                            self.weights[i, j, :] += learning_rate * influence * (input_vec - self.weights[i, j, :])

                # linear normalization constraint
                for r in range(self.input_dim):
                    N_r = np.sum(self.weights[:, :, r])  
                    sum = np.sum(self.weights[:, :, r], axis=(0, 1))  
                    if sum > 0:
                        self.weights[:, :, r] = N_r * self.weights[:, :, r] / sum


    def find_bmu(self, input_vec):
        bmu_idx = np.argmin(np.linalg.norm(self.weights - input_vec, axis=-1))
        return np.unravel_index(bmu_idx, (self.height, self.width))
    
    def plot_som(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        half_dim = self.input_dim // 2  
        # input vector is divided in two parts; first half is one eye
        for i in range(self.height):
            for j in range(self.width):
                right_eye_weight = np.sum(self.weights[i, j, :half_dim])
                left_eye_weight = np.sum(self.weights[i, j, half_dim:])
                total_weight = right_eye_weight + left_eye_weight
                if right_eye_weight > 0.5 * total_weight:
                    color = 'black'  # dominant for one eye
                elif left_eye_weight > 0.5 * total_weight:
                    color = 'white'  # dominant for other eye
                else:
                    color = 'gray'  # dominant for neither eye
                    
                ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1, facecolor=color, edgecolor='none'))
        
        plt.show()

# helper function to generate inputs
def generate_processed_inputs(num_samples, sigma_r=1.5, h=0.15):
    inputs = []
    for _ in range(num_samples):
        eye1 = np.random.choice([0.,1.], size=(16, 16))
        eye2 = np.random.choice([0.,1.], size=(16, 16))
        
        # convolve eyes with gaussian function of standard deviation sigma_r
        eye1_convolved = gaussian_filter(eye1, sigma=sigma_r)
        eye2_convolved = gaussian_filter(eye2, sigma=sigma_r)
        
        # adjust activity according to ha_j + (1-h)a_j'
        adjusted_eye1 = h * eye1_convolved + (1 - h) * eye2_convolved
        adjusted_eye2 = h * eye2_convolved + (1 - h) * eye1_convolved
        
        # flatten and concatenate the adjusted convolved eyes to form the input vector
        input_vec = np.concatenate([adjusted_eye1.flatten(), adjusted_eye2.flatten()])
        inputs.append(input_vec)
      
    return np.array(inputs)

# generate inputs
num_samples = 10
inputs = generate_processed_inputs(num_samples)

# initialize SOM
som = SOM(32, 32, 16*16*2) 
som.train(inputs, num_epochs=1000)
som.plot_som()