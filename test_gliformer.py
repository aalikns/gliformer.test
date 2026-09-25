import torch
from gliformer import GLiFormer

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Kullanılacak cihaz: {device}")
print("Model yükleniyor...")

model = GLiFormer.from_pretrained(
    "knowledgator/gliformer-large-v1",
    load_tokenizer=True,
)
model = model.to(device).eval()
print("Model hazır! Çıkmak için metin yerine 'q' yaz.\n")

while True:
    text = input("Metin gir: ")
    if text.lower() == "q":
        break

    labels_input = input("Etiketler (virgülle ayır, örn: person,organization,location): ")
    labels = [label.strip() for label in labels_input.split(",")]

    threshold_input = input("Threshold (boş bırakırsan 0.5): ").strip()
    threshold = float(threshold_input) if threshold_input else 0.5

    entities = model.predict_entities(
        text,
        labels,
        threshold=threshold,
    )

    if entities:
        for entity in entities:
            print(f"  {entity['text']} → {entity['label']} ({entity['score']:.4f})")
    else:
        print("  Hiçbir varlık bulunamadı.")
    print()
