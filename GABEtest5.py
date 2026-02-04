# Copyright (c) 2026 Dmitry Feklin
# Apache License 2.0

import torch
import torch.nn as nn
import torch.optim as optim
import timm
from torchvision.models import resnet18, ResNet18_Weights
from GABE import GABE
from collections import defaultdict
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def group_model_weights(model: nn.Module, layer_types: list):
    groups = defaultdict(list)
    for module in model.modules():
        if isinstance(module, tuple(layer_types)):
            weight = module.weight.detach().clone()
            if isinstance(module, nn.Conv2d):
                weight = weight.view(weight.shape[0], -1)
            elif isinstance(module, nn.Linear):
                pass
            else:
                continue
            groups[weight.shape].append(weight)
    return {shape: tensors for shape, tensors in groups.items() if len(tensors) > 1}

def extract_coeffs(model: nn.Module, layer_types: list):
    compressor = GABE()
    weight_groups = group_model_weights(model, layer_types)
    coeffs_dict = {}
    for shape, weights_list in weight_groups.items():
        compressed = compressor.compress(weights_list, basis_rank=1, w_bar_rank=16)
        coeffs_dict[shape] = compressed["coeffs"]
    return coeffs_dict

def compute_correlations(coeffs_dicts: dict):
    correlations = {}
    shapes = set(coeffs_dicts[list(coeffs_dicts.keys())[0]].keys())
    for shape in shapes:
        coeff_vectors = []
        model_names = []
        for model_name, cdict in coeffs_dicts.items():
            if shape in cdict:
                coeff_vectors.append(cdict[shape].flatten())
                model_names.append(model_name)
        if len(coeff_vectors) > 1:
            stack = torch.stack(coeff_vectors)
            corr_matrix = torch.corrcoef(stack.float())
            correlations[shape] = (model_names, corr_matrix)
    return correlations

def identify_stable_layers(correlations: dict, threshold: float = 0.9):
    stable_shapes = []
    for shape, (_, corr_matrix) in correlations.items():
        corr = corr_matrix[0,1].item()
        if corr >= threshold:
            stable_shapes.append(shape)
    return stable_shapes

# -----------------------------
# MLP generator
# -----------------------------

class CoeffGenerator(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# -----------------------------
# Collecting data for the generator
# -----------------------------

def prepare_training_data(model, stable_shapes, unstable_shapes, layer_types, num_batches=50, batch_size=8, device='cpu'):
    X_list = []
    y_list = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, 3, 32, 32).to(device)
        coeffs = extract_coeffs(model, layer_types)
        # Stable layers
        stable_vec = []
        for s in stable_shapes:
            stable_vec.append(coeffs[s].flatten())
        X_list.append(torch.cat(stable_vec))
        # Unstable layers
        unstable_vec = []
        for u in unstable_shapes:
            unstable_vec.append(coeffs[u].flatten())
        y_list.append(torch.cat(unstable_vec))
    X = torch.stack(X_list)  # (num_batches, total_stable_coeffs)
    y = torch.stack(y_list)  # (num_batches, total_unstable_coeffs)
    return X, y

# -----------------------------
# Visualization of generator prediction
# -----------------------------

def plot_prediction(y_true, y_pred, title="Prediction vs Original"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true.flatten().detach().numpy(), y_pred.flatten().detach().numpy(), alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Original coefficients")
    plt.ylabel("Predicted coefficients")
    plt.title(title)
    plt.grid(True)
    plt.show()

# -----------------------------
# Main script
# -----------------------------

def mlp_generator_skill_transfer():
    device = "cpu"
    torch.manual_seed(42)

    # Loading models
    model_source = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    model_target = timm.create_model("hf_hub:SamAdamDay/resnet18_cifar10", pretrained=True).to(device)
    layer_types = [nn.Conv2d, nn.Linear]

    # Extracting coefficients to determine stable layers
    source_coeffs = extract_coeffs(model_source, layer_types)
    target_coeffs = extract_coeffs(model_target, layer_types)
    coeffs_dicts = {"source": source_coeffs, "target": target_coeffs}
    
    correlations = compute_correlations(coeffs_dicts)
    stable_shapes = identify_stable_layers(correlations, threshold=0.9)
    unstable_shapes = [s for s in target_coeffs.keys() if s not in stable_shapes]

    print("Stable layers:", stable_shapes)
    print("Unstable layers:", unstable_shapes)

    # -----------------------------
    # Data preparation
    # -----------------------------
    X_train, y_train = prepare_training_data(model_target, stable_shapes, unstable_shapes, layer_types,
                                             num_batches=100, batch_size=8, device=device)

    input_dim = X_train.shape[1]
    output_dim = y_train.shape[1]

    # -----------------------------
    # Generator training
    # -----------------------------
    gen = CoeffGenerator(input_dim, output_dim).to(device)
    optimizer = optim.Adam(gen.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    epochs = 300
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = gen(X_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss={loss.item():.6f}")

    # -----------------------------
    # Checking the generator
    # -----------------------------
    with torch.no_grad():
        y_pred_final = gen(X_train)
        r2 = r2_score(y_train.numpy(), y_pred_final.numpy())
        print(f"\nFinal R² for unstable layers prediction: {r2:.4f}")

        # Visualization
        plot_prediction(y_train, y_pred_final, title="MLP Generator Prediction vs Original Coefficients")

# -----------------------------

if __name__ == "__main__":
    mlp_generator_skill_transfer()
