import os
import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# ওয়েব পেজের ডিজাইন ও টাইটেল
st.set_page_config(
    page_title='Lung Cancer Detection', page_icon='🩺', layout='centered'
)

st.title('🩺 Lung Cancer Detection (Normal vs Malignant)')
st.write(
    'This web application uses a PyTorch Deep Learning model (ResNet18) to'
    ' classify chest CT scans as Normal or Malignant.'
)

# ক্লাস নেম ডিফাইন (এখানে শুধু ২টি ক্লাস)
class_names = ['Malignant', 'Normal']  # ফোল্ডারের অ্যালফাবেটিকাল ক্রম অনুযায়ী

# ইমেজ ট্রান্সফর্মেশন
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# মডেল লোড করার ফাংশন
@st.cache_resource
def load_model():
  model = models.resnet18(weights=None)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(
      num_ftrs, 2
  )  # ক্লাস সংখ্যা ২ সেট করা হলো (Normal ও Malignant)

  if os.path.exists('lung_cancer_classifier.pth'):
    model.load_state_dict(
        torch.load(
            'lung_cancer_classifier.pth', map_location=torch.device('cpu')
        )
    )
  model.eval()
  return model

model = load_model()

# সাইডবারে মডেল ট্রেনিংয়ের অপশন
st.sidebar.header('Model Training Panel')
if st.sidebar.button('Train Model Now'):
  with st.spinner('Training is in progress... Please wait.'):
    try:
      # ডেটাসেট লোড
      image_datasets = datasets.ImageFolder('dataset', transform)
      dataloader = DataLoader(image_datasets, batch_size=32, shuffle=True)

      # ট্রেনিং সেটআপ
      train_model = models.resnet18(weights='DEFAULT')
      train_model.fc = nn.Linear(
          train_model.fc.in_features, 2
      )  # এখানেও আউটপুট লেয়ার ২ দিতে হবে
      criterion = nn.CrossEntropyLoss()
      optimizer = optim.Adam(train_model.parameters(), lr=0.001)

      train_model.train()
      num_epochs = 3  # ইপক সংখ্যা

      for epoch in range(num_epochs):
        for inputs, labels in dataloader:
          optimizer.zero_grad()
          outputs = train_model(inputs)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

      # ওয়েটস সেভ করা
      torch.save(train_model.state_dict(), 'lung_cancer_classifier.pth')
      st.sidebar.success(
          'Model trained successfully! Please refresh the page.'
      )
    except Exception as e:
      st.sidebar.error(
          f"Error: Make sure 'dataset' folder with Normal and Malignant exists."
          f' Details: {e}'
      )

# মূল ইন্টারফেস: সিটি স্ক্যান ছবি আপলোড করার জায়গা
st.subheader('Upload CT Scan Image')
uploaded_file = st.file_uploader(
    'Choose an image...', type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  st.image(image, caption='Uploaded CT Scan Image', use_container_width=True)

  if st.button('Predict'):
    with st.spinner('Analyzing the scan...'):
      img_tensor = transform(image).unsqueeze(0)

      with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, 0)

      predicted_class = class_names[predicted.item()]
      confidence_score = confidence.item() * 100

      # ফলাফল প্রদর্শন
      st.markdown('---')
      st.subheader('Prediction Result:')
      if predicted_class == 'Normal':
        st.success(
            f'**Result:** {predicted_class} (Confidence:'
            f' {confidence_score:.2f}%)'
        )
      else:
        st.error(
            f'**Result:** {predicted_class} - Cancerous/Malignant detected!'
            f' (Confidence: {confidence_score:.2f}%)'
        )