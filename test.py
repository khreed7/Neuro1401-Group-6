import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class SOM:
    #initializing the grid of neurons
    def __init__(self, height, width, input_dim):
        #height and weight are represented by "r" in the article for inputs, and "c" for outputs
        self.height = height
        self.width = width
        self.input_dim = input_dim
        #random variable synaptic weights associated with each neuron
        self.weights = np.random.random((height, width, input_dim))

    def train(self, inputs, num_epochs=100, initial_learning_rate=0.1, initial_radius=None):
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
                    sumofweights = 0
                    sumofweights = sumofweights + self.weights[i,:,:]
                    for j in range(self.width):
                        dist_to_bmu = np.linalg.norm(np.array([i,j]) - np.array(bmu_idx))
                        if dist_to_bmu <= radius:
                            influence = np.exp(-dist_to_bmu**2 / (2 * (radius**2)))
                            #implementing hebbian to update weights at each time-step
                            self.weights[i,j,:] = self.weights[i,j,:] + learning_rate * input_vec * influence * (input_vec - self.weights[i,j,:])
                    #implementing subtractive normalization for each cortical unit

                    #NOTE: commenting this out because it works but produces all white output, which i think will be fixed once gaussian is fixed
                    #self.weights[i,:,:] = self.weights[i,:,:] / sumofweights
                                
    
    def find_bmu(self, input_vec):
        #implementing linear summation where self is represented by w_cr in article and inputs is a_r in article

        #NOTE: test code, as i'm not sure this is completely right integration w/bmu but it runs
        #bmu_idx = np.argmin(np.linalg.norm(self.weights * input_vec, axis=-1))
        #return np.unravel_index(bmu_idx, (self.height, self.width))

        #NOTE: original code:
        bmu_idx = np.argmin(np.linalg.norm(self.weights - input_vec, axis=-1))
        return np.unravel_index(bmu_idx, (self.height, self.width))
    
    def plot_som(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        
        for i in range(self.height):
            for j in range(self.width):
                grayscale_value = np.mean(self.weights[i, j])
                color = (grayscale_value, grayscale_value, grayscale_value)
                ax.add_patch(plt.Rectangle((j, self.height-1-i), 1, 1, facecolor=color, edgecolor='none'))
        
        plt.show()

# helper function to generate inputs
def generate_processed_inputs(num_samples, sigma_r=1.5, h=0.5):
    inputs = []
    for _ in range(num_samples):
        eye1 = np.random.choice([0, 1], size=(16, 16))
        eye2 = np.random.choice([0, 1], size=(16, 16))
        
        # convolve eyes with gaussian function of standard deviation sigma_r
        #eye1_convolved = gaussian_filter(eye1, sigma=sigma_r)
        #eye2_convolved = gaussian_filter(eye2, sigma=sigma_r)
        
        # adjust activity according to ha_j + (1-h)a_j'
        adjusted_eye1 = h * eye1 + (1 - h) * eye2
        adjusted_eye2 = h * eye2 + (1 - h) * eye1
        #adjusted_eye1 = h * eye1_convolved + (1 - h) * eye2_convolved
        #adjusted_eye2 = h * eye2_convolved + (1 - h) * eye1_convolved
        
        # flatten and concatenate the adjusted convolved eyes to form the input vector
        input_vec = np.concatenate([adjusted_eye1.flatten(), adjusted_eye2.flatten()])
        inputs.append(input_vec)
    return np.array(inputs)

# generate inputs
num_samples = 10
inputs = generate_processed_inputs(num_samples)

# initialize SOM
som = SOM(16, 16, 16*16*2)  
som.train(inputs, num_epochs=100)
som.plot_som()
