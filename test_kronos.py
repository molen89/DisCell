from kronos import create_model, create_model_from_pretrained
model, precision, embedding_dim = create_model_from_pretrained(
    checkpoint_path="hf_hub:MahmoodLab/kronos", # Make sure you have requested access on HuggingFace
    cache_dir="./model_assets",
)

print("Model precision: ", precision)
print("Model embedding dimension: ", embedding_dim)