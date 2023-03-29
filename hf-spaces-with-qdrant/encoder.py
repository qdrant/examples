import clip

model, _ = clip.load("ViT-B/32")
device = next(model.parameters()).device


def encode_text(text: str):
    return model.encode_text(clip.tokenize([text]).to(device)).detach().cpu().numpy().squeeze().tolist()
