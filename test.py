from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dino-vitb8")
print(model)