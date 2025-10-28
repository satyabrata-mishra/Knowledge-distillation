import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm  # For progress bars
import time
import copy # To save the best model

# --- 1. Setup ---
# Set device (use GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Hyperparameters ---
# You can experiment with these values for your report
TEACHER_EPOCHS = 15       # Epochs to train the teacher
STUDENT_EPOCHS = 20       # Epochs to train both student models
BATCH_SIZE = 128
LEARNING_RATE = 0.001
TEMPERATURE = 4.0         # Temperature for distillation
ALPHA = 0.7               # Weight for the soft loss (distillation)
                          # (1 - ALPHA) will be the weight for the hard loss (standard cross-entropy)

# --- 2. Model Definitions ---

# Teacher Model (A large, pre-trained model)
# We use ResNet-18 from torchvision
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Load a ResNet-18 model pre-trained on ImageNet
        self.model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        
        # Modify the final fully-connected layer for CIFAR-10 (10 classes)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 10)

    def forward(self, x):
        return self.model(x)

# Student Model (A compact, lightweight CNN)
class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # A simple CNN architecture
        # Input images are 3x32x32 (Channels x Height x Width)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        # -> 16x32x32
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 16x16x16
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        # -> 32x16x16
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # -> 32x8x8
        
        # Flatten the output for the fully-connected layers
        # 32 channels * 8 height * 8 width = 2048
        self.fc1 = nn.Linear(32 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10) # 10 output classes for CIFAR-10

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Data Loading (CIFAR-10) ---

# Define transformations for the data
# For training: add data augmentation (random crops, horizontal flips)
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# For testing: only normalize
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Download and load the CIFAR-10 datasets
trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(
    trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(
    testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 4. Training & Evaluation Functions ---

# Standard training loop (for Teacher and Conventional Student)
def train_model(model, trainloader, optimizer, criterion, epoch):
    model.train() # Set model to training mode
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}")
    
    for inputs, labels in progress_bar:
        # Move data to the selected device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        # Update progress bar
        progress_bar.set_postfix(loss=running_loss/len(progress_bar))

# Evaluation loop (used by all models)
def evaluate_model(model, testloader):
    model.eval() # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad(): # No gradients needed for evaluation
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

# Special Knowledge Distillation (KD) training loop
def train_distilled_student(teacher, student, trainloader, optimizer, epoch, T, alpha):
    teacher.eval() # Teacher is frozen and in eval mode
    student.train() # Student is in training mode
    
    running_loss = 0.0
    progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1} (Distill)")
    
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # --- This is the core of Knowledge Distillation ---
        
        # 1. Get Teacher's predictions (logits)
        # We don't need gradients for the teacher
        with torch.no_grad():
            teacher_logits = teacher(inputs)
        
        # 2. Get Student's predictions (logits)
        student_logits = student(inputs)
        
        # 3. Calculate the "soft loss" (Distillation Loss)
        # This compares the student's soft predictions to the teacher's soft predictions
        # It uses the Temperature (T)
        soft_loss = nn.KLDivLoss(reduction='batchmean')(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits / T, dim=1)
        ) * (T * T) # Scale the loss back up
        
        # 4. Calculate the "hard loss" (Standard Training Loss)
        # This compares the student's normal predictions to the true labels
        hard_loss = F.cross_entropy(student_logits, labels)
        
        # 5. Combine the losses using alpha
        # total_loss = (alpha * soft_loss) + ((1 - alpha) * hard_loss)
        total_loss = (alpha * soft_loss) + ((1.0 - alpha) * hard_loss)
        
        # Backward pass and optimize (updates only the student)
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        progress_bar.set_postfix(loss=running_loss/len(progress_bar))


# --- 5. Utility Functions for Metrics ---

# Function to count the number of trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to measure average inference speed per image
def measure_inference_speed(model, testloader):
    model.eval()
    total_time = 0.0
    num_images = 0
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.to(device)
            
            start_time = time.time()
            _ = model(inputs)
            # Ensure operation is complete for accurate timing on GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            end_time = time.time()
            
            total_time += (end_time - start_time)
            num_images += labels.size(0)
            
    return total_time / num_images # Average time per image

# --- 6. Main Execution ---

# !--- THIS IS THE FIX ---!
# All the code that *runs* the project must be inside this block.
if __name__ == '__main__':

    # Dictionary to store final results
    results = {}

    # --- Phase 1: Train the Teacher Model ---
    print("\n--- Phase 1: Training Teacher Model (ResNet-18) ---")
    teacher_model = TeacherModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=LEARNING_RATE)

    best_teacher_acc = 0.0
    # We save a copy of the best performing teacher model
    best_teacher_model_state = None

    for epoch in range(TEACHER_EPOCHS):
        train_model(teacher_model, trainloader, optimizer_teacher, criterion, epoch)
        acc = evaluate_model(teacher_model, testloader)
        print(f"Teacher Accuracy after Epoch {epoch+1}: {acc:.2f}%")
        
        if acc > best_teacher_acc:
            best_teacher_acc = acc
            # Save the model's state dictionary
            best_teacher_model_state = copy.deepcopy(teacher_model.state_dict())

    # Load the best performing teacher model for final metrics
    teacher_model.load_state_dict(best_teacher_model_state)
    teacher_acc = best_teacher_acc
    teacher_params = count_parameters(teacher_model)
    teacher_speed = measure_inference_speed(teacher_model, testloader)
    results['Teacher'] = (teacher_acc, teacher_params, teacher_speed)
    print(f"Final Teacher Accuracy: {teacher_acc:.2f}%")

    # --- Phase 2: Train the Student Model (Conventionally) ---
    print("\n--- Phase 2: Training Student Model (Conventional) ---")
    student_conventional = StudentModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_student_conv = optim.Adam(student_conventional.parameters(), lr=LEARNING_RATE)

    best_student_conv_acc = 0.0
    for epoch in range(STUDENT_EPOCHS):
        train_model(student_conventional, trainloader, optimizer_student_conv, criterion, epoch)
        acc = evaluate_model(student_conventional, testloader)
        print(f"Conventional Student Accuracy after Epoch {epoch+1}: {acc:.2f}%")
        if acc > best_student_conv_acc:
            best_student_conv_acc = acc

    student_conv_acc = best_student_conv_acc
    student_params = count_parameters(student_conventional) # Same for both students
    student_conv_speed = measure_inference_speed(student_conventional, testloader)
    results['Student (Conventional)'] = (student_conv_acc, student_params, student_conv_speed)
    print(f"Final Conventional Student Accuracy: {student_conv_acc:.2f}%")


    # --- Phase 3: Train the Distilled Student Model ---
    print("\n--- Phase 3: Training Student Model (Distilled) ---")
    # Create a new, untrained student model
    student_distilled = StudentModel().to(device)
    optimizer_student_distill = optim.Adam(student_distilled.parameters(), lr=LEARNING_RATE)

    best_student_distill_acc = 0.0
    for epoch in range(STUDENT_EPOCHS):
        # Use the special distillation training function
        train_distilled_student(
            teacher=teacher_model,
            student=student_distilled,
            trainloader=trainloader,
            optimizer=optimizer_student_distill,
            epoch=epoch,
            T=TEMPERATURE,
            alpha=ALPHA
        )
        acc = evaluate_model(student_distilled, testloader)
        print(f"Distilled Student Accuracy after Epoch {epoch+1}: {acc:.2f}%")
        if acc > best_student_distill_acc:
            best_student_distill_acc = acc

    student_distill_acc = best_student_distill_acc
    student_distill_speed = measure_inference_speed(student_distilled, testloader)
    results['Student (Distilled)'] = (student_distill_acc, student_params, student_distill_speed)
    print(f"Final Distilled Student Accuracy: {student_distill_acc:.2f}%")


    # --- Phase 4: Comparative Performance Analysis ---
    print("\n" + "="*80)
    print(f"{'Final Project Results: Comparative Performance Analysis':^80}")
    print("="*80)
    print(f"{'Model':<25} | {'Accuracy (%)':<15} | {'Parameters':<15} | {'Avg. Inference (s/img)':<25}")
    print("-"*80)

    # Teacher
    model_name = "1. Teacher Model"
    acc, params, speed = results['Teacher']
    print(f"{model_name:<25} | {acc:<15.2f} | {params:<15,} | {speed:<25.8f}")

    # Conventional Student
    model_name = "2. Student (Conventional)"
    acc, params, speed = results['Student (Conventional)']
    print(f"{model_name:<25} | {acc:<15.2f} | {params:<15,} | {speed:<25.8f}")

    # Distilled Student
    model_name = "3. Student (Distilled)"
    acc, params, speed = results['Student (Distilled)']
    print(f"{model_name:<25} | {acc:<15.2f} | {params:<15,} | {speed:<25.8f}")

    print("="*80)

    # --- Success Criterion Check ---
    print("\nSuccess Criterion Check:")
    distill_acc = results['Student (Distilled)'][0]
    conv_acc = results['Student (Conventional)'][0]
    if distill_acc > conv_acc:
        print(f"✅ Success: Distilled Student ({distill_acc:.2f}%) achieved higher accuracy than Conventional Student ({conv_acc:.2f}%).")
    else:
        print(f"❌ Inconclusive: Distilled Student ({distill_acc:.2f}%) did not outperform Conventional Student ({conv_acc:.2f}%).")
        print("   (Try tuning Epochs, Learning Rate, Alpha, or Temperature for better results).")
