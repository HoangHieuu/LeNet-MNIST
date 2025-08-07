import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as transforms

class LeNetClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding='same')
        self.avgpool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avgpool2 = nn.AvgPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.fc_1 = nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = nn.Linear(120, 84)
        self.fc_3 = nn.Linear(84, num_classes)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.avgpool1(outputs)
        outputs = F.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.avgpool2(outputs)
        outputs = F.relu(outputs)
        outputs = self.flatten(outputs)
        outputs = self.fc_1(outputs)
        outputs = self.fc_2(outputs)
        outputs = self.fc_3(outputs)
        return outputs

@st.cache_resource
def load_model(model_path, num_classes=10):
    lenet_model = LeNetClassifier(num_classes)
    lenet_model.load_state_dict(torch.load(model_path, weights_only=True, map_location=torch.device('cpu')))
    lenet_model.eval()
    return lenet_model
model = load_model('lenet_model.pt')

def inference(image, model):
    w, h = image.size
    if w != h:
        crop = transforms.CenterCrop(min(w, h))
        image = crop(image)
        wnew, hnew = image.size
    img_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])
    img_new = img_transform(image)
    img_new = img_new.expand(1, 1, 28, 28)
    with torch.no_grad():
        predictions = model(img_new)
    preds = nn.Softmax(dim=1)(predictions)
    p_max, yhat = torch.max(preds.data, 1)
    return p_max.item()*100, yhat.item()

def main():
    # 1. Page config
    st.set_page_config(
        page_title="Digit Recognition",
        page_icon="🔢",
        layout="centered"
    )

    # 2. Ẩn menu & footer mặc định
    st.markdown(
        """
        <style>
          #MainMenu {visibility: hidden;}
          footer {visibility: hidden;}
          header {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )

    # 3. Tiêu đề chính
    st.markdown("<h1 style='text-align: center; color: #00edfb;'>Digit Recognition</h1>",
                unsafe_allow_html=True)
    st.markdown("#### Model: **LeNet**  &nbsp;•&nbsp; Dataset: **MNIST**", unsafe_allow_html=True)

    # 4. Sidebar chọn nguồn ảnh
    st.sidebar.header("Input Options")
    choice = st.sidebar.radio("", ["Upload Your Image", "Run Example"])

    # 5. Xử lý Upload
    if choice == "Upload Your Image":
        uploaded = st.sidebar.file_uploader(
            "Upload a digit image (PNG/JPG)", type=["png", "jpg", "jpeg"]
        )
        if uploaded:
            img = Image.open(uploaded).convert("L")
            st.sidebar.image(img, caption="Preview", width=120)

            # **Dùng inference()** để dự đoán
            prob, pred = inference(img, model)

            # Hiển thị song song
            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Your Input", use_container_width=True)
            with col2:
                st.markdown("### Prediction")
                st.metric(label="Digit", value=f"{pred}")
                st.markdown(f"**Confidence:** {prob:.2f} %")
                st.progress(int(prob))

    # 6. Xử lý Example
    else:
        if st.sidebar.button("Run Example"):
            img = Image.open("demo_0.jpg").convert("L")
            st.sidebar.image(img, caption="Demo", width=120)

            # **Dùng inference()** để dự đoán
            prob, pred = inference(img, model)

            col1, col2 = st.columns(2)
            with col1:
                st.image(img, caption="Example Input", use_container_width=True)
            with col2:
                st.markdown("### Prediction")
                st.metric(label="Digit", value=f"{pred}")
                st.markdown(f"**Confidence:** {prob:.2f} %")
                st.progress(int(prob))

    # 7. Footer nhỏ
    st.markdown(
        """
        ---
        <div style='text-align:center; font-size:12px; color:gray;'>
          Built using LeNet & Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------
# Entry point
# -----------------------
if __name__ == "__main__":
    main()