# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 데이터 경로 설정
data_dir = ".\data"
train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
val_dir = os.path.join(data_dir, "Validation")

# 이미지 전처리 변환
input_size = 224
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 사용자 정의 데이터셋
class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, subfolder in enumerate(["fake", "real"]):
            class_folder = os.path.join(folder, subfolder)
            for image_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, image_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)


class ImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for label, subfolder in enumerate(["fake", "real"]):
            class_folder = os.path.join(folder, subfolder)
            for image_name in os.listdir(class_folder):
                self.image_paths.append(os.path.join(class_folder, image_name))
                self.labels.append(label)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = cv2.imread(img_path)  # OpenCV로 이미지 읽기
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR을 RGB로 변환
        img = Image.fromarray(img)  # numpy -> PIL 이미지로 변환
        if self.transform:
            img = self.transform(img)  # transforms 적용
        return img, torch.tensor(label, dtype=torch.float32)

# 데이터 로드
train_dataset = ImageDataset(train_dir, transform=transform)
val_dataset = ImageDataset(val_dir, transform=transform)
test_dataset = ImageDataset(test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ResNeXt 모델 정의
class MyResNeXt(nn.Module):
    def __init__(self):
        super(MyResNeXt, self).__init__()
        base_model = resnet50(pretrained=True)
        base_model.fc = nn.Linear(base_model.fc.in_features, 1)  # 이진 분류를 위한 출력
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# Xception 모델 정의
class MyXception(nn.Module):
    def __init__(self):
        super(MyXception, self).__init__()
        base_model = ptcv_get_model("xception", pretrained=True)
        base_model.output = nn.Linear(base_model.output.in_features, 1)
        self.model = base_model

    def forward(self, x):
        return self.model(x)

# ResNeXt 또는 Xception 선택
use_xception = False  # True로 변경하면 Xception 모델 사용
if use_xception:
    model = MyXception().to(device)
else:
    model = MyResNeXt().to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 학습 함수
def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# 평가 함수
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device).unsqueeze(1)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return running_loss / len(dataloader), accuracy

# 학습 및 평가 루프
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = train_model(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")

# 테스트 데이터에서 평가
test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

# 모델 저장
torch.save(model.state_dict(), "deepfake_image_model.pth")
print("Model saved to 'deepfake_image_model.pth'")