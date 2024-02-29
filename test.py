import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class SOM:
    def __init__(self, height, width, input_dim):
        self.height = height
        self.width = width
        self.input_dim = input_dim
        self.weights = np.random.random((height, width, input_dim))
        self.beta = beta

        # Notionally scale the retinal sheet to the size of the cortical sheet
        retinal_height = retinal_width = int(np.sqrt(input_dim // 2))  # Assuming input_dim represents two retinal sheets
        self.retinal_positions = np.dstack(np.meshgrid(np.linspace(0, retinal_height-1, retinal_height), 
                                                       np.linspace(0, retinal_width-1, retinal_width), 
                                                       indexing='ij'))

        # Calculate normalized distances for the initial bias
        self.initial_weights = np.zeros((self.height, self.width, self.input_dim))
        for i in range(self.height):
            for j in range(self.width):
                for k in range(retinal_height):
                    for l in range(retinal_width):
                        cortical_distance = np.array([i/self.height, j/self.width])
                        retinal_distance = np.array([k/retinal_height, l/retinal_width])
                        distance = np.linalg.norm(cortical_distance - retinal_distance)
                        normalized_distance = distance / np.sqrt(2)  # Normalize by the maximum possible distance
                        
                        # Set initial weights based on the normalized distance and beta
                        weight_range = self.beta * normalized_distance
                        self.initial_weights[i, j, k*retinal_width + l] = np.random.uniform(low=0, high=weight_range)
                        self.initial_weights[i, j, input_dim//2 + k*retinal_width + l] = np.random.uniform(low=0, high=weight_range)

        self.weights = self.initial_weights.copy()

    def train(self, inputs, num_epochs, initial_learning_rate=0.01, initial_radius=None):
        if initial_radius is None:
            initial_radius = max(self.height, self.width) / 2
        time_constant = num_epochs / np.log(initial_radius)

        neuron_positions = np.array(np.meshgrid(np.arange(self.height), np.arange(self.width), indexing='ij')).transpose(1, 2, 0)
        
        for epoch in range(num_epochs):
            learning_rate = initial_learning_rate * np.exp(-epoch / num_epochs)
            radius = initial_radius * np.exp(-epoch / time_constant)
            radius_sq = radius ** 2

            for input_vec in inputs:
                bmu_idx = self.find_bmu(input_vec)
                dist_sq_to_bmu = np.sum((neuron_positions - np.array(bmu_idx)) ** 2, axis=2)

                influence = np.exp(-dist_sq_to_bmu / (2 * radius_sq))
                influence[dist_sq_to_bmu > radius_sq] = 0

                # vectorized weight update for all dimensions
                delta = learning_rate * (input_vec - self.weights) * influence[:, :, np.newaxis]
                self.weights += delta

            # vectorized linear normalization constraint
            for r in range(self.input_dim):
                total_weight = np.sum(self.weights[:, :, r])
                if total_weight > 0:
                    self.weights[:, :, r] /= total_weight / np.sum(self.weights[:, :, r], axis=(0, 1))

    def find_bmu(self, input_vec):
        dist_sq = np.sum((self.weights - input_vec) ** 2, axis=2)
        return np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    
    def plot_som(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        half_dim = self.input_dim // 2  
        # input vector is divided in two parts; first half is one eye
        for i in range(self.height):
            for j in range(self.width):
                left_eye_weight = np.sum(self.weights[i, j, :half_dim])
                right_eye_weight = np.sum(self.weights[i, j, half_dim:])
                total_weight = left_eye_weight + right_eye_weight
                if left_eye_weight > 0.5 * total_weight:
                    color = 'black'  # dominant for one eye
                elif right_eye_weight > 0.5 * total_weight:
                    color = 'white'  # dominant for other eye
                else:
                    color = 'gray'  # dominant for neither eye
                    
                ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1, facecolor=color, edgecolor='none'))
        
        plt.show()

    def plot_weights_in_retinal_space(self):
        retinal_height, retinal_width = self.retinal_positions.shape[:2]
        half_dim = self.input_dim // 2  # Assuming input_dim represents two retinal sheets
        retinal_shape = (retinal_height, retinal_width)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Calculate and plot the center of mass for the weights associated with each cortical unit
        for i in range(self.height):
            for j in range(self.width):
                weights_left_eye = self.weights[i, j, :half_dim].reshape(retinal_shape)
                weights_right_eye = self.weights[i, j, half_dim:].reshape(retinal_shape)
                
                # Calculate centers of mass
                y_left, x_left = np.unravel_index(np.argmax(weights_left_eye), weights_left_eye.shape)
                y_right, x_right = np.unravel_index(np.argmax(weights_right_eye), weights_right_eye.shape)
                
                # Plot centers of mass
                ax.plot(x_left, retinal_height-1-y_left, 'bo', label='Left Eye' if i == 0 and j == 0 else "")
                ax.plot(x_right, retinal_height-1-y_right, 'ro', label='Right Eye' if i == 0 and j == 0 else "")

        ax.set_xlim(0, retinal_width-1)
        ax.set_ylim(0, retinal_height-1)
        ax.set_title('Centers of Mass of Weights in Retinal Space')
        ax.set_xlabel('Retinal Width')
        ax.set_ylabel('Retinal Height')
        ax.legend()

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
num_samples = 1000
inputs = generate_processed_inputs(num_samples)
beta = 0.5

# initialize SOM
som = SOM(32, 32, 16*16*2) 
som.train(inputs, num_epochs=1)
som.plot_som()
som.plot_weights_in_retinal_space()