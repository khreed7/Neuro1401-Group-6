import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

def hsv_to_rgb(h, s, v):
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
    return np.array([r, g, b])

def calculate_contrast(palette):
    luminances = [0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2] for row in palette for color in row]
    max_luminance = max(luminances)
    min_luminance = min(luminances)
    return (max_luminance + 0.05) / (min_luminance + 0.05)

def calculate_color_variability(palette):
    flattened_palette = palette.reshape(-1, 3)
    return np.std(flattened_palette, axis=0).mean()

def generate_color_palette(similar=True, dissimilar_hue_variation=180):
    base_hue = np.random.rand() * 360
    hue_variation = 15 if similar else dissimilar_hue_variation
    palette = np.zeros((3, 3, 3))
    for i in range(3):
        for j in range(3):
            hue = (base_hue + np.random.randn() * hue_variation) % 360
            saturation = 0.7 + 0.3 * np.random.rand()
            value = 0.7 + 0.3 * np.random.rand()
            rgb = hsv_to_rgb(hue, saturation, value)
            palette[i, j] = rgb
    contrast = calculate_contrast(palette)
    variability = calculate_color_variability(palette)
    return palette, contrast, variability

class ColorChangeDataset(Dataset):
    def __init__(self, size=1000, change_probability=0.5, mix_ratio=0.5):
        self.data = []
        for _ in range(size):
            similar = np.random.rand() < mix_ratio
            base_palette, contrast, variability = generate_color_palette(similar)
            after_palette = np.copy(base_palette)
            changed = False
            if np.random.rand() < change_probability:
                i, j = np.random.randint(0, 3), np.random.randint(0, 3)
                new_hue = (base_palette[i, j, 0] * 360 + 180) % 360 # edit this 
                after_palette[i, j] = hsv_to_rgb(new_hue, base_palette[i, j, 1], base_palette[i, j, 2])
                changed = True
            self.data.append((base_palette, after_palette, changed, contrast, variability))

    def __getitem__(self, idx):
        before, after, changed, contrast, variability = self.data[idx]
        before = torch.tensor(before, dtype=torch.float32).permute(2, 0, 1)
        after = torch.tensor(after, dtype=torch.float32).permute(2, 0, 1)
        return before, after, torch.tensor(changed, dtype=torch.long), contrast, variability

    def __len__(self):
        return len(self.data)

class ColorChangeDetectorCNN(nn.Module):
    def __init__(self):
        super(ColorChangeDetectorCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 16, kernel_size=(3, 3), padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=1, stride=1)
        self.fc1 = nn.Linear(16 * 3 * 3, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, before, after):
        x = torch.cat((before, after), dim=1)
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model, dataset, epochs=64, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for data in loader:
            before, after, labels, contrast, variability = data 
            optimizer.zero_grad()
            outputs = model(before, after)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

def test_model(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    TP = 0 
    TN = 0 
    FP = 0 
    FN = 0  
    
    model.eval()
    with torch.no_grad():
        for data in loader:
            before, after, labels, contrast, variability = data
            outputs = model(before, after)
            _, predicted = torch.max(outputs, 1)
            
            TP += ((predicted == 1) & (labels == 1)).sum().item()
            TN += ((predicted == 0) & (labels == 0)).sum().item()
            FP += ((predicted == 1) & (labels == 0)).sum().item()
            FN += ((predicted == 0) & (labels == 1)).sum().item()

    total = TP + TN + FP + FN
    accuracy = 100 * (TP + TN) / total if total > 0 else 0
    same = 100 * TP / (TP + FP) if (TP + FP) > 0 else 0
    different = 100 * TN / (TN + FN) if (TN + FN) > 0 else 0
    return accuracy, same, different, TP, TN, FP, FN

dataset_train = ColorChangeDataset()
model = ColorChangeDetectorCNN()
train_model(model, dataset_train)

dataset_similar = ColorChangeDataset(size=1000, change_probability=0.5, mix_ratio=1)
dataset_different = ColorChangeDataset(size=1000, change_probability=0.5, mix_ratio=0)

accuracy_similar, same_similar, different_similar, TP_similar, TN_similar, FP_similar, FN_similar = test_model(model, dataset_similar)
accuracy_different, same_different, different_different, TP_different, TN_different, FP_different, FN_different = test_model(model, dataset_different)

print(f"Results for similar palettes:")
print(f"Accuracy: {accuracy_similar}%, TP rate: {same_similar}%, TN rate: {different_similar}%")
print(f"TP: {TP_similar}, TN: {TN_similar}, FP: {FP_similar}, FN: {FN_similar}")

print(f"Results for different palettes:")
print(f"Accuracy: {accuracy_different}%, TP rate: {same_different}%, TN rate: {different_different}%")
print(f"TP: {TP_different}, TN: {TN_different}, FP: {FP_different}, FN: {FN_different}")

'''
Plot palettes
'''

'''
def plot_palette_comparison(before_palette, after_palette, title="Palette Comparison"):
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    axs[0].imshow(before_palette, aspect='auto')
    axs[0].set_title('Before')
    axs[0].axis('off')

    axs[1].imshow(after_palette, aspect='auto')
    axs[1].set_title('After')
    axs[1].axis('off')

    plt.suptitle(title)
    plt.show()

dataset_similar = ColorChangeDataset(size=1, change_probability=1, mix_ratio=1)  
dataset_different = ColorChangeDataset(size=1, change_probability=1, mix_ratio=0)  

before_similar, after_similar, _, _, _ = dataset_similar[0]
before_different, after_different, _, _, _ = dataset_different[0]

plot_palette_comparison(before_similar.numpy().transpose(1, 2, 0), after_similar.numpy().transpose(1, 2, 0), title="Similar Palettes")
plot_palette_comparison(before_different.numpy().transpose(1, 2, 0), after_different.numpy().transpose(1, 2, 0), title="Dissimilar Palettes")
'''