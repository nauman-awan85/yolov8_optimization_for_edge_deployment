import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

model_path = r"C:\Users\Muhammad Nauman\Desktop\new_thesis\models\new model\finetune.pt"

output_path = model_path.replace(".pt", "_pruned.pt")

prune_ratio = 0.30  # prune 30% of filters per Conv layer

device = "cuda" if torch.cuda.is_available() else "cpu"

yolo_model = YOLO(model_path)
model = yolo_model.model.to(device).eval()


# module_container_map, dictionary mapping each submodule (Conv, BN, etc.) to its container block.
module_container_map = {}

def build_module_map(container):

    for _, sublayer in container.named_children():
        module_container_map[sublayer] = container
        build_module_map(sublayer)
        
build_module_map(model)

# func to find corresponding batchnorm layer
def get_associated_bn(container_block: nn.Module, conv_layer: nn.Conv2d):

    if container_block is None:
        return None

    bn_layer = getattr(container_block, "bn", None)
    if isinstance(bn_layer, nn.BatchNorm2d) and bn_layer.num_features == conv_layer.out_channels:
        return bn_layer

    for attr_name in dir(container_block):
        candidate = getattr(container_block, attr_name, None)
        if isinstance(candidate, nn.BatchNorm2d) and candidate.num_features == conv_layer.out_channels:
            return candidate

    return None

# Pruning

total_layers = 0
total_filters = 0
pruned_filters = 0

skip_first_conv = False

with torch.no_grad():
    for layer in model.modules():
        if isinstance(layer, Detect):
            continue

        if isinstance(layer, nn.Conv2d) and layer.weight is not None:
            if not skip_first_conv:
                skip_first_conv = True
                continue
# Extract weights of this convolutional layer
            weights = layer.weight
            num_filters = weights.shape[0]
            if num_filters <= 1:
                continue

# Determine number of filters to prune based on ratio
            num_to_prune = int(round(prune_ratio * num_filters))
            if num_to_prune <= 0:
                continue
            if num_to_prune >= num_filters:
                num_to_prune = num_filters - 1

# Compute filter importance scores 
            scores = weights.abs().sum(dim=(1, 2, 3))
            prune_indices = torch.topk(scores, num_to_prune, largest=False).indices

#making least imp weights to zero
            weights[prune_indices, ...] = 0.0

# Locate and zero relevant BatchNorm parameters
            bn_layer = get_associated_bn(module_container_map.get(layer), layer)
            if isinstance(bn_layer, nn.BatchNorm2d):
                bn_layer.weight[prune_indices] = 0.0
                if bn_layer.bias is not None:
                    bn_layer.bias[prune_indices] = 0.0
                if hasattr(bn_layer, "running_mean"):
                    bn_layer.running_mean[prune_indices] = 0.0
                if hasattr(bn_layer, "running_var"):
                    bn_layer.running_var[prune_indices] = 1.0

            total_layers += 1
            total_filters += num_filters
            pruned_filters += num_to_prune

print(f"Pruned {pruned_filters}/{total_filters} filters (~{100 * pruned_filters / max(1, total_filters):.1f}%)")


# Sparsity calculation

element_total = element_zero = filter_total = filter_zero = 0

with torch.no_grad():
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) and layer.weight is not None:
            weights = layer.weight
            element_total += weights.numel()
            element_zero += (weights == 0).sum().item()
            num_filters = weights.shape[0]
            filter_total += num_filters
            filter_zero += (weights.view(num_filters, -1).sum(dim=1) == 0).sum().item()

print(f"Element sparsity: {100.0 * element_zero / max(1, element_total):.2f}%")
print(f"Filter sparsity : {100.0 * filter_zero / max(1, filter_total):.2f}% ({filter_zero}/{filter_total})")


yolo_model.model = model.eval()
yolo_model.save(output_path)
print(f"Saved: {output_path}")
