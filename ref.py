# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader, random_split


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load the dataset
# dataset = datasets.ImageFolder(root='data', transform=transform)

# # Split dataset into training and validation sets
# train_size = int(0.7 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create data loaders for training and validation sets
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # Define the model architecture
# class ImageClassifier(nn.Module):
#     def __init__(self):
#         super(ImageClassifier, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.fc1 = nn.Linear(64*54*54, 128)
#         self.fc2 = nn.Linear(128, 1)

#     def forward(self, x):
#         x = nn.functional.relu(self.conv1(x))
#         x = nn.functional.max_pool2d(x, 2, 2)
#         x = nn.functional.relu(self.conv2(x))
#         x = nn.functional.max_pool2d(x, 2, 2)
#         x = x.view(-1, 64*54*54)
#         x = nn.functional.relu(self.fc1(x))
#         x = torch.sigmoid(self.fc2(x))
#         return x


# model = ImageClassifier().to(device)
# criterion = nn.BCELoss() 
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# for epoch in range(50):
#     model.train()
#     running_loss = 0.0
#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data[0].to(device), data[1].float().to(device)  # Ensure labels are float for BCELoss

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueeze labels to match output shape
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0

#     # Validation loop
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     val_correct = 0
#     with torch.no_grad():
#         for inputs, labels in val_loader:
#             inputs, labels = inputs.to(device), labels.float().to(device)
#             outputs = model(inputs)
#             val_loss += criterion(outputs, labels.unsqueeze(1)).item()
#             predicted = torch.round(outputs)
#             val_correct += (predicted == labels.unsqueeze(1)).sum().item()

#     val_loss /= len(val_loader.dataset)
#     val_accuracy = val_correct / len(val_loader.dataset)
#     print('Validation Loss: {:.4f}, Accuracy: {:.4f}'.format(val_loss, val_accuracy))

# # Save the trained model
# torch.save(model.state_dict(), 'ai_generated_model.pth')
# print('Model saved as "ai_generated_model.pth"')

# print('Finished Training')



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import datasets, transforms, models
# from torch.utils.data import DataLoader, random_split

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# ])

# # Load the dataset
# dataset = datasets.ImageFolder(root='data', transform=transform)

# # Split dataset into training and validation sets
# train_size = int(0.7 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create data loaders for training and validation sets
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# # Define the model architecture using a pre-trained ResNet
# class ImageClassifier(nn.Module):
#     def __init__(self):
#         super(ImageClassifier, self).__init__()
#         self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)
    
#     def forward(self, x):
#         x = self.base_model(x)
#         return torch.sigmoid(x)

# model = ImageClassifier().to(device)
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Learning rate scheduler
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# # Training loop
# for epoch in range(50):
#     model.train()
#     running_loss = 0.0
#     running_corrects = 0
#     total_train = 0

#     for i, data in enumerate(train_loader, 0):
#         inputs, labels = data[0].to(device), data[1].float().to(device)  # Ensure labels are float for BCELoss

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueeze labels to match output shape
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         predicted = torch.round(outputs)
#         running_corrects += (predicted == labels.unsqueeze(1)).sum().item()
#         total_train += labels.size(0)

#         if i % 100 == 99:
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 100))
#             running_loss = 0.0

#     train_accuracy = running_corrects / total_train
#     print('Epoch [{}], Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch + 1, running_loss, train_accuracy))

#     # Step the scheduler
#     scheduler.step()

# # Evaluate the model on the validation set after training is complete
# model.eval()
# val_loss = 0.0
# val_corrects = 0
# total_val = 0
# with torch.no_grad():
#     for inputs, labels in val_loader:
#         inputs, labels = inputs.to(device), labels.float().to(device)
#         outputs = model(inputs)
#         loss = criterion(outputs, labels.unsqueeze(1))  # Unsqueeze labels to match output shape
#         val_loss += loss.item()
#         predicted = torch.round(outputs)
#         val_corrects += (predicted == labels.unsqueeze(1)).sum().item()
#         total_val += labels.size(0)

# val_loss /= len(val_loader)  # Average over batches
# val_accuracy = val_corrects / total_val
# print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(val_loss, val_accuracy))

# # Save the trained model
# torch.save(model.state_dict(), 'improved_model.pth')
# print('Model saved as "improved_model.pth"')

# print('Finished Training')


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the dataset
dataset = datasets.ImageFolder(root='data', transform=transform)

# Split dataset into training and validation sets
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Define the model architecture using a pre-trained ResNet
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, 1)
    
    def forward(self, x):
        x = self.base_model(x)
        return torch.sigmoid(x)

model = ImageClassifier().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Training loop
for epoch in range(50):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_train = 0

    for i, data in enumerate(train_loader, 0):
        # Ensure labels are float for BCELoss
        inputs, labels = data[0].to(device), data[1].float().to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        # Unsqueeze labels to match output shape
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = torch.round(outputs)
        running_corrects += (predicted == labels.unsqueeze(1)).sum().item()
        total_train += labels.size(0)

        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

    train_accuracy = running_corrects / total_train
    print('Epoch [{}], Training Loss: {:.4f}, Training Accuracy: {:.4f}'.format(epoch + 1, running_loss, train_accuracy))

    # Step the scheduler
    scheduler.step()

# Evaluate the model on the validation set after training is complete
model.eval()
val_loss = 0.0
val_corrects = 0
total_val = 0
all_labels = []
all_outputs = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.float().to(device)
        outputs = model(inputs)
        # Unsqueeze labels to match output shape
        loss = criterion(outputs, labels.unsqueeze(1))
        val_loss += loss.item()
        predicted = torch.round(outputs)
        val_corrects += (predicted == labels.unsqueeze(1)).sum().item()
        total_val += labels.size(0)
        all_labels.extend(labels.cpu().numpy())
        all_outputs.extend(outputs.cpu().numpy())

# Average over batches
val_loss /= len(val_loader)
val_accuracy = val_corrects / total_val
val_auc = roc_auc_score(all_labels, all_outputs)
print('Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, Validation AUC: {:.4f}'.format(val_loss, val_accuracy, val_auc))

# Save the trained model
torch.save(model.state_dict(), 'improved_model.pth')
print('Model saved as "improved_model.pth"')

print('Finished Training')