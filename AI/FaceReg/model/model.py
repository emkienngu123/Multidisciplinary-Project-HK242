import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from facenet_pytorch import MTCNN
import numpy as np

class FaceReg(nn.Module):
    def __init__(self, cfg):
        super(FaceReg, self).__init__()
        self.cfg = cfg
        self.embedding_size = cfg['model']['embedding_size']
        self.mtcnn = MTCNN()

        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.eval()

        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # MobileNetV3 expects 224x224 input
            transforms.ToTensor()
        ])

        self.mlp = nn.Sequential(
            nn.Linear(1024, self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size)
        )
    def detect_faces(self, image):
        try:
            boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
            return boxes, probs, landmarks
        except Exception as e:
            print(f"Error during face detection: {e}")
            return None, None, None
    def extract_face(self, image, box):
        try:
            face = image.crop(box)
            return face
        except Exception as e:
            print(f"Error during face extraction: {e}")
            return None
    def align_face(self, face_image, landmarks):
        if landmarks is not None and len(landmarks) > 0:
            left_eye = landmarks[0]
            right_eye = landmarks[1]

            # Calculate the angle between the eyes
            delta_x = right_eye[0] - left_eye[0]
            delta_y = right_eye[1] - left_eye[1]
            angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

            # Perform rotation
            rotated_face = face_image.rotate(angle, center=((left_eye[0] + right_eye[0]) / 2,
                                                            (left_eye[1] + right_eye[1]) / 2))
            return rotated_face
        return face_image
    def extract_base_features(self, face_image):
        try:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            img_tensor = self.preprocess(face_image).unsqueeze(0).to(device)
            with torch.no_grad():
                features = self.backbone(img_tensor)
                # Global average pooling to get a feature vector
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                flattened_features = torch.flatten(features, 1)
            return flattened_features
        except Exception as e:
            print(f"Error during base feature extraction: {e}")
            return None
    def extract_features(self, face_image):
        base_features = self.extract_base_features(face_image)
        if base_features is not None:
            return self.mlp(base_features)
        else:
            return None
    def get_face_embedding(self, image):
        boxes, _, landmarks = self.detect_faces(image)

        if boxes is not None and len(boxes) > 0:
            # Assuming we want to process the first detected face
            first_box = [int(b) for b in boxes[0]]
            first_landmarks = landmarks[0] if landmarks is not None else None
            cropped_face = self.extract_face(image, first_box)
            if cropped_face is not None:
                aligned_face = self.align_face(cropped_face, first_landmarks)
                embedding = self.extract_features(aligned_face)
                if embedding is not None:
                    return embedding
        return None
    def forward(self, image):
        """
        Main forward pass for the module.  Simplifies the usage.  This version
        returns the output *after* the MLP.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
             numpy.ndarray or None:  The face embedding, or None on failure.
        """
        return self.get_face_embedding(image)
def build_model(cfg):
    """
    Build the face recognition model based on the provided configuration.

    Args:
        cfg (dict): Configuration dictionary containing model parameters.

    Returns:
        FaceReg: An instance of the FaceReg model.
    """
    return FaceReg(cfg)
# class FaceFeatureExtractor(nn.Module):
#     """
#     This module combines MTCNN face detection with MobileNetV3-Small feature extraction,
#     and includes an MLP for training with triplet loss.
#     """
#     def __init__(self, device='cpu', embedding_size=128):
#         super(FaceFeatureExtractor, self).__init__()
#         self.device = device
#         self.embedding_size = embedding_size # Size of the final embedding vector

#         # MTCNN for face detection
#         self.mtcnn = MTCNN(device=self.device)

#         # MobileNetV3-Small for feature extraction
#         self.mobilenet_v3_small = models.mobilenet_v3_small(pretrained=True)
#         self.mobilenet_v3_small.eval()  # Set to evaluation mode for feature extraction
#         # Remove the classifier layers to get the feature extractor
#         self.feature_extractor = nn.Sequential(*list(self.mobilenet_v3_small.children())[:-1])

#         # Define the image transformations.  Important to do this *once*
#         self.preprocess = transforms.Compose([
#             transforms.Resize((224, 224)),  # MobileNetV3 expects 224x224 input
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ])

#         # MLP for training with triplet loss
#         self.mlp = nn.Sequential(
#             nn.Linear(1024, 512),  # Adjust input size if needed
#             nn.ReLU(),
#             nn.Linear(512, self.embedding_size)
#         ).to(self.device)

#     def detect_faces(self, image_path):
#         """Detects faces in an image using MTCNN.

#         Args:
#             image_path (str): Path to the image file.

#         Returns:
#             tuple: A tuple containing:
#                 - list: A list of bounding boxes (as lists of [x1, y1, x2, y2]).
#                 - list: A list of detection probabilities/confidences.
#                 - list: A list of facial landmarks (as lists of (x, y) coordinates
#                         for each landmark point).
#                 Returns (None, None, None) if no faces are detected.
#         """
#         try:
#             img = Image.open(image_path).convert('RGB')
#             boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
#             return boxes, probs, landmarks
#         except Exception as e:
#             print(f"Error during face detection: {e}")
#             return None, None, None

#     def extract_face(self, image_path, box):
#         """Extracts a detected face region from the image based on the bounding box.

#         Args:
#             image_path (str): Path to the image file.
#             box (list): Bounding box coordinates [x1, y1, x2, y2].

#         Returns:
#             PIL.Image.Image or None:  The cropped face image, or None if error.
#         """
#         try:
#             img = Image.open(image_path).convert('RGB')
#             face = img.crop(box)
#             return face
#         except Exception as e:
#             print(f"Error during face extraction: {e}")
#             return None

#     def align_face(self, face_image, landmarks):
#         """Aligns a detected face based on facial landmarks using affine transformation.

#         Args:
#             face_image (PIL.Image.Image): The cropped face image.
#             landmarks (list): Facial landmarks (list of (x, y) coordinates).

#         Returns:
#             PIL.Image.Image: The aligned face image, or the original if alignment fails.
#         """
#         if landmarks is not None and len(landmarks) > 0:
#             left_eye = landmarks[0]
#             right_eye = landmarks[1]

#             # Calculate the angle between the eyes
#             delta_x = right_eye[0] - left_eye[0]
#             delta_y = right_eye[1] - left_eye[1]
#             angle = np.arctan2(delta_y, delta_x) * 180 / np.pi

#             # Perform rotation
#             rotated_face = face_image.rotate(angle, center=((left_eye[0] + right_eye[0]) / 2,
#                                                             (left_eye[1] + right_eye[1]) / 2))
#             return rotated_face
#         return face_image

#     def extract_base_features(self, face_image):
#         """Extracts feature embeddings from a face image using MobileNetV3-Small (before MLP).

#         Args:
#             face_image (PIL.Image.Image): The preprocessed and resized face image.

#         Returns:
#             torch.Tensor: The feature vector extracted by MobileNetV3-Small.
#         """
#         try:
#             img_tensor = self.preprocess(face_image).unsqueeze(0).to(self.device)
#             with torch.no_grad():
#                 features = self.feature_extractor(img_tensor)
#                 # Global average pooling to get a feature vector
#                 features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
#                 flattened_features = torch.flatten(features, 1)
#             return flattened_features
#         except Exception as e:
#             print(f"Error during base feature extraction: {e}")
#             return None

#     def extract_features(self, face_image):
#         """Extracts final feature embeddings from a face image using MobileNetV3-Small and MLP.

#         Args:
#             face_image (PIL.Image.Image): The preprocessed and resized face image.

#         Returns:
#             torch.Tensor: The final feature vector after passing through MLP.
#         """
#         base_features = self.extract_base_features(face_image)
#         if base_features is not None:
#             return self.mlp(base_features)
#         else:
#             return None

#     def get_face_embedding(self, image_path):
#         """Detects faces in an image, extracts the first detected face,
#         aligns it, and then extracts its feature embedding.

#         Args:
#             image_path (str): Path to the image file.

#         Returns:
#             numpy.ndarray or None: The feature embedding of the detected face,
#                                    or None if no face is detected or extraction fails.
#         """
#         boxes, _, landmarks = self.detect_faces(image_path)

#         if boxes is not None and len(boxes) > 0:
#             # Assuming we want to process the first detected face
#             first_box = [int(b) for b in boxes[0]]
#             first_landmarks = landmarks[0] if landmarks is not None else None
#             cropped_face = self.extract_face(image_path, first_box)
#             if cropped_face is not None:
#                 aligned_face = self.align_face(cropped_face, first_landmarks)
#                 embedding = self.extract_features(aligned_face)
#                 if embedding is not None:
#                   return embedding.cpu().numpy()
#         return None

#     def forward(self, image_path):
#         """
#         Main forward pass for the module.  Simplifies the usage.  This version
#         returns the output *after* the MLP.

#         Args:
#             image_path (str): Path to the input image.

#         Returns:
#              numpy.ndarray or None:  The face embedding, or None on failure.
#         """
#         return self.get_face_embedding(image_path)

# class TripletLoss(nn.Module):
#     """
#     Triplet loss implementation.
#     """
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin

#     def forward(self, anchor, positive, negative):
#         """
#         Forward pass for the triplet loss.

#         Args:
#             anchor (torch.Tensor): Anchor embeddings.
#             positive (torch.Tensor): Positive embeddings.
#             negative (torch.Tensor): Negative embeddings.

#         Returns:
#             torch.Tensor: The triplet loss.
#         """
#         dist_pos = torch.sum((anchor - positive) ** 2, dim=1)
#         dist_neg = torch.sum((anchor - negative) ** 2, dim=1)
#         loss = torch.relu(dist_pos - dist_neg + self.margin)
#         return torch.mean(loss)

# class FaceDataset(Dataset):
#     """
#     Dataset class for loading face image triplets.
#     """
#     def __init__(self, image_paths, labels, transform=None):
#         """
#         Args:
#             image_paths (list): List of image file paths.
#             labels (list): List of corresponding labels (IDs).
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform
#         self.label_to_indices = self._get_label_to_indices()

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, index):
#         """
#         Gets a triplet (anchor, positive, negative) based on the given index.
#         """
#         anchor_path = self.image_paths[index]
#         anchor_label = self.labels[index]

#         positive_index = np.random.choice(self.label_to_indices[anchor_label])
#         while positive_index == index:
#             positive_index = np.random.choice(self.label_to_indices[anchor_label])
#         positive_path = self.image_paths[positive_index]

#         negative_label = np.random.choice([label for label in self.label_to_indices.keys() if label != anchor_label])
#         negative_index = np.random.choice(self.label_to_indices[negative_label])
#         negative_path = self.image_paths[negative_index]

#         anchor_img = Image.open(anchor_path).convert('RGB')
#         positive_img = Image.open(positive_path).convert('RGB')
#         negative_img = Image.open(negative_path).convert('RGB')


#         if self.transform:
#             anchor_img = self.transform(anchor_img)
#             positive_img = self.transform(positive_img)
#             negative_img = self.transform(negative_img)

#         return anchor_img, positive_img, negative_img, \
#                anchor_path, positive_path, negative_path # Return paths for feature extraction

#     def _get_label_to_indices(self):
#         """Helper function to create a dictionary mapping labels to indices."""
#         label_to_indices = {}
#         for i, label in enumerate(self.labels):
#             if label not in label_to_indices:
#                 label_to_indices[label] = []
#             label_to_indices[label].append(i)
#         return label_to_indices

# def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
#     """
#     Trains the face recognition model using triplet loss.

#     Args:
#         model (FaceFeatureExtractor): The face feature extractor model.
#         train_loader (DataLoader): DataLoader for the training data.
#         criterion (nn.Module): The triplet loss function.
#         optimizer (optim.Optimizer): The optimizer.
#         num_epochs (int, optional): Number of epochs to train. Default is 10.
#     """
#     model.train()  # Set the model to training mode
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for i, (anchor_images, positive_images, negative_images, anchor_paths, positive_paths, negative_paths) in enumerate(train_loader):

#             #anchor_images, positive_images, negative_images = anchor_images.to(device), positive_images.to(device), negative_images.to(device)

#             # Pass images through the model to get embeddings
#             anchor_embeddings = []
#             positive_embeddings = []
#             negative_embeddings = []

#             for anchor_path, positive_path, negative_path in zip(anchor_paths, positive_paths, negative_paths):
#                 anchor_embedding = model.get_face_embedding(anchor_path)
#                 positive_embedding = model.get_face_embedding(positive_path)
#                 negative_embedding = model.get_face_embedding(negative_path)

#                 if anchor_embedding is not None and positive_embedding is not None and negative_embedding is not None:
#                     anchor_embeddings.append(anchor_embedding)
#                     positive_embeddings.append(positive_embedding)
#                     negative_embeddings.append(negative_embedding)

#             if not anchor_embeddings or not positive_embeddings or not negative_embeddings:
#                 print(f"Skipping batch {i} due to missing embeddings.")
#                 continue  # Skip this batch if any embeddings are missing

#             anchor_embeddings = torch.tensor(np.array(anchor_embeddings)).to(model.device)
#             positive_embeddings = torch.tensor(np.array(positive_embeddings)).to(model.device)
#             negative_embeddings = torch.tensor(np.array(negative_embeddings)).to(model.device)

#             # Zero the gradients
#             optimizer.zero_grad()

#             # Calculate the triplet loss
#             loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
#             # Backpropagate the loss
#             loss.backward()
#             # Update the weights
#             optimizer.step()

#             running_loss += loss.item()
#             if i % 10 == 9:    # print every 10 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 10))
#                 running_loss = 0.0

#     print('Finished Training')
#     model.eval() # Set model to eval after training
#     return model

# def main():
#     """
#     Main function to run the face feature extraction and training.
#     """
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     embedding_size = 128
#     face_extractor = FaceFeatureExtractor(device, embedding_size).to(device)

#     # Example data loading
#     image_paths = ['your_image_1.jpg', 'your_image_2.jpg', 'your_image_3.jpg', 'your_image_4.jpg', 'your_image_5.jpg', 'your_image_6.jpg']  # Replace with your image paths
#     labels = [0, 0, 0, 1, 1, 1]  # Example labels (0 for person 1, 1 for person 2)
#     # Create a dataset
#     face_dataset = FaceDataset(image_paths, labels, transform=transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#         ]))
#     # Create a dataloader
#     train_loader = DataLoader(face_dataset, batch_size=2, shuffle=True) # Adjust batch size as needed

#     # Define the loss function and optimizer
#     criterion = TripletLoss(margin=1.0)  # You can adjust the margin
#     optimizer = optim.Adam(face_extractor.mlp.parameters(), lr=0.001)  # Optimize *only* the MLP

#     # Train the model
#     trained_model = train_model(face_extractor, train_loader, criterion, optimizer, num_epochs=10)  # Adjust the number of epochs

#     # Example of getting an embedding after training
#     test_image_path = 'your_test_image.jpg' # Replace with path
#     test_embedding = trained_model(test_image_path) # Use the forward pass
#     if test_embedding is not None:
#         print("Test Face embedding shape:", test_embedding.shape)
#         print("Test Face embedding:", test_embedding)
#     else:
#         print("No face detected or feature extraction failed.")
    
#     # Save the trained model
#     torch.save(trained_model.state_dict(), 'face_recognition_model.pth')
#     print("Trained model saved to face_recognition_model.pth")

# if __name__ == '__main__':
#     main()
