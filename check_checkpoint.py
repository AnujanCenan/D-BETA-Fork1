import torch
checkpoint = torch.load("checkpoints/best.pt", map_location="cpu")
keys = list(checkpoint["model"].keys())
print(f"Total keys: {len(keys)}")
print("First 20 keys:")
for k in keys[:20]:
    print(" ", k)
print("Last 10 keys:")
for k in keys[-10:]:
    print(" ", k)

# Check specifically for feature extractor keys
fe_keys = [k for k in keys if "feature_extractor" in k]
print(f"Feature extractor keys: {len(fe_keys)}")
for k in fe_keys:
    print(" ", k)

# Also check conv keys generally
conv_keys = [k for k in keys if "conv" in k and "ecg_encoder" in k]
print(f"\nECG encoder conv keys: {len(conv_keys)}")
for k in conv_keys:
    print(" ", k)
