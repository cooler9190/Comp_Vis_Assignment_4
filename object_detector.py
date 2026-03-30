from torch import nn

class ObjectDetector(nn.Module):
    def __init__(self):
        super().__init__()

        # Block 1: Input(112, 112, 3) -> Conv(16, 112, 112) -> Pool(16, 56, 56)
        # Using 3x3 kernels, padding=1, max pooling better for image classification and object detection tasks
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # YOLO paper uses MaxPool.

        # Block 2: Input(16, 56, 56) -> Conv(32, 56, 56) -> Pool(32, 28, 28)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # YOLO paper uses MaxPool.

        # Block 3: Input(32, 28, 28) -> Conv(64, 28, 28) -> Pool(64, 14, 14)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # YOLO paper uses MaxPool.

        # Block 4: Input(64, 14, 14) -> Conv(64, 14, 14) -> Pool(64, 7, 7)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # YOLO paper uses MaxPool.

        # Block 5: Input(64, 7, 7) -> Conv(32, 7, 7)
        # No pooling layer according to assignment specs
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(32)

        # Flatten layer to convert 3D feature maps to 1D feature vector for fully connected layers
        # 32 channels * 7 height * 7 width = 1568 features
        self.flatten = nn.Flatten()

        # Dropout
        dropout_probability = 0.5
        self.dropout = nn.Dropout(p=dropout_probability)

        # Fully connected layers: 512 neurons
        self.fc1 = nn.Linear(in_features=32*7*7, out_features=512)
        
        # Output layer: 343 neurons 
        self.fc_out = nn.Linear(in_features=512, out_features=343)
        self.out_activation = nn.Sigmoid()  # Sigmoid activation for output layer to get values in [0, 1]

        # Reusable ReLU activation for hidden layers
        self.relu = nn.LeakyReLU(negative_slope=0.1) # From YOLO paper.
        # self.relu = nn.ReLU() # Simpler ReLU activation function

        # Apply custom weight initialization
        self._initialize_weights()
    
    def forward(self, x):
        # Apply Conv -> BatchNorm -> ReLU -> Pooling for each block
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(self.conv4(x))))

        # Block 5 has no pooling layer
        x = self.relu(self.bn5(self.conv5(x)))

        x = self.flatten(x)
        x = self.dropout(x)  # Apply dropout before fully connected layers

        x = self.relu(self.fc1(x))
        x = self.out_activation(self.fc_out(x))  # Apply sigmoid activation to output layer

        return x
    
    def _initialize_weights(self):
        # Iterate through all modules in the network
        for m in self.modules():
            # Apply Kaiming Uniform initialization to convolutional and linear layers
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                # 'nonlinearity' tells PyTorch to optimize the gain for ReLU
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                # Initialize biases to zero if they exist
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
