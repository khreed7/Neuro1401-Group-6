import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

def hsv_to_rgb(h, s, v):
    """ 
    Convert hsv to rgb 
    """
    h = float(h) / 360
    s = float(s)
    v = float(v)
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
    return [r, g, b]


def generate_color_palette(similar=True):
    """ 
    !! - needs editing to reflect paper

    Generate similar and different 3 x 3 color palettes. 
    Similar palettes have less variation (hue within 15 degrees),
    different palettes have more variation (within 180 degrees).
    """
    base_hue = np.random.rand() * 360
    hue_variation = 15 if similar else 180
    palette = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            hue = (base_hue + np.random.randn() * hue_variation) % 360
            saturation = 0.7 + 0.3 * np.random.rand()
            value = 0.7 + 0.3 * np.random.rand()
            rgb = hsv_to_rgb(hue, saturation, value)
            palette[i, j] = rgb
    return palette

class ColorPaletteDataset(Dataset):
    """
    Generate color palettes and flag if similar (1 = similar, 0 = dissimilar)
    """
    def __init__(self, size=1000, proportion_similar=0.7):  
        num_similar = int(size * proportion_similar)
        num_dissimilar = size - num_similar
        self.data = [generate_color_palette(True) for _ in range(num_similar)] + \
                    [generate_color_palette(False) for _ in range(num_dissimilar)]
        self.labels = np.random.randint(0, 2, size)
        self.similarity_flags = [1] * num_similar + [0] * num_dissimilar

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = torch.tensor(self.data[idx], dtype=torch.float32).permute(2, 0, 1)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        similarity_flag = torch.tensor(self.similarity_flags[idx], dtype=torch.float32)
        return sample, label, similarity_flag

class ColorMemoryCNN(nn.Module):
    """
    Convolution neural network
    """
    def __init__(self):
        super(ColorMemoryCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x, similarity_flags):
        inhibition_factor = (1.0 - 0.5 * similarity_flags.unsqueeze(1).unsqueeze(2).unsqueeze(3))
        x = x * inhibition_factor
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def custom_loss(outputs, labels, similarity_flags, base_loss_fn, scale_factor=5.0):
    """
    Adjust loss based on similarity. 
    Increases loss for similar palettes (similarity_flags = 1)
    """
    base_loss = base_loss_fn(outputs, labels)
    scaled_loss = base_loss * (1 + scale_factor * similarity_flags) 
    return scaled_loss.mean()

def train_test_model(dataset_train, dataset_test, model, epochs=20, batch_size=10):
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        for inputs, labels, similarity_flags in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs, similarity_flags)
            loss = custom_loss(outputs, labels, similarity_flags, criterion)
            loss.backward()
            optimizer.step()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, similarity_flags in test_loader:
            outputs = model(inputs, similarity_flags)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

dataset_train = ColorPaletteDataset(size=1000, proportion_similar=0.75)
dataset_similar_test = ColorPaletteDataset(size=1000, proportion_similar=1)
dataset_dissimilar_test = ColorPaletteDataset(size=1000, proportion_similar=0)

model = ColorMemoryCNN()
print("Training model on mixed dataset...")
train_test_model(dataset_train, dataset_train, model)

accuracy_similar = train_test_model(dataset_train, dataset_similar_test, model)
accuracy_dissimilar = train_test_model(dataset_train, dataset_dissimilar_test, model)

print(f"Accuracy on test set with similar hues: {accuracy_similar}%")
print(f"Accuracy on test set with dissimilar hues: {accuracy_dissimilar}%")