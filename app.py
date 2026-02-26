import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os

# -------- Settings --------
IMAGE_FOLDER = "images"
EMBED_FILE = "embeddings.pt"

st.title("🔍 AI Image Search Engine")

# -------- Load model --------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()

# -------- Build embeddings --------
def build_embeddings():
    vectors = []
    paths = []

    for file in os.listdir(IMAGE_FOLDER):
        if file.lower().endswith((".jpg",".jpeg",".png")):
            path = os.path.join(IMAGE_FOLDER,file)
            image = Image.open(path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")

            with torch.no_grad():
                feats = model.get_image_features(pixel_values=inputs["pixel_values"])
                feats = feats / feats.norm(dim=-1, keepdim=True)

            vectors.append(feats)
            paths.append(path)

    if not vectors:
        return None

    vectors = torch.cat(vectors)
    torch.save({"vectors":vectors,"paths":paths}, EMBED_FILE)

    return vectors, paths

# -------- Load or build --------
if os.path.exists(EMBED_FILE):
    data = torch.load(EMBED_FILE)
    image_vectors = data["vectors"]
    image_paths = data["paths"]
else:
    result = build_embeddings()
    if result:
        image_vectors, image_paths = result
    else:
        st.error("No images found on Desktop")
        st.stop()

# -------- Search UI --------
query = st.text_input("Search images using text")
uploaded_file = st.file_uploader("Or upload an image", type=["jpg","jpeg","png"])

if st.button("Search") and query:

    inputs = processor(text=[query], return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    scores = (image_vectors @ text_features.T).squeeze()
    top5 = torch.topk(scores, k=min(5, len(scores)))

    st.subheader("Top Matches")

    cols = st.columns(3)

    for i, idx in enumerate(top5.indices):
        score = scores[idx].item()
        similarity = round(score * 100, 2)

        with cols[i % 3]:
            st.image(image_paths[idx], use_container_width=True)
            st.markdown(
                f"""
                <div style="
                    background:#111;
                    padding:10px;
                    border-radius:10px;
                    text-align:center;
                    margin-bottom:15px;
                ">
                    <b>Similarity:</b> {similarity}%
                </div>
                """,
                unsafe_allow_html=True
            )
        # -------- Reverse image search --------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", width=300)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=inputs["pixel_values"]
        )

    image_vec = image_features / image_features.norm(dim=-1, keepdim=True)

    scores = (image_vectors @ image_vec.T).squeeze()
    top5 = torch.topk(scores, k=min(5, len(scores)))

    st.subheader("Best Matches from uploaded image:")

    for idx in top5.indices:

        st.image(image_paths[idx], width=300)


