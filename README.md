# CS5720_CNN
##Nikhila Jajapuram
##700759188

##Question 2: Convolution Operations with Different Parameters
Here we conduct convolution operations on a 5×5 input matrix using a 3×3 kernel with varying stride and padding values. The implementation uses NumPy for matrix manipulation and TensorFlow/Keras for convolution operations.

###Steps:

###Define the Input Matrix
A 5×5 matrix is initialized as a NumPy array.
It is reshaped to (1, 5, 5, 1) to match the input format required by TensorFlow’s Conv2D layer.
###Define the Kernel (Filter)
A 3×3 kernel (Laplacian-like filter for edge detection) is defined.
It is reshaped to (3, 3, 1, 1) to match the expected format of convolution filters.
###Function to Apply Convolution
Uses a Sequential model with a single Conv2D layer.
The kernel is initialized using tf.constant_initializer(kernel), ensuring the defined kernel is used.
The function applies convolution using the specified stride and padding.
###Perform Convolutions with Different Parameters
The convolution operation is tested with four combinations:
  - Stride = 1, Padding = 'VALID' → No padding, keeps only valid outputs.
  - Stride = 1, Padding = 'SAME' → Adds padding to maintain the same size.
  - Stride = 2, Padding = 'VALID' → No padding, moves the kernel with a step of 2.
  - Stride = 2, Padding = 'SAME' → Adds padding to maintain dimensions.
###Print the Output Feature Maps
The code here prints the output feature maps for each convolution setting.


##Question 3: CNN Feature Extraction with Filters and Pooling
The code uses CNN feature extraction using edge detection (Sobel filter) and pooling operations (Max and Average Pooling).

###Task 1: Edge Detection using Sobel Filter
Loads an image in grayscale using OpenCV (cv2.imread()).
Defines Sobel filters for detecting edges in the x and y directions.
Applies the Sobel filters using cv2.filter2D().
Displays:
- Original image
- Edge-detected image using Sobel-X
- Edge-detected image using Sobel-Y

###Task 2: Max Pooling and Average Pooling
Creates a random 4×4 matrix to simulate an image.
Uses MaxPooling2D (selects the highest value in a 2×2 region).
Uses AveragePooling2D (computes the average of values in a 2×2 region).
Prints the original, max-pooled, and average-pooled matrices.


##Question 4: Implementing and Comparing CNN Architectures 
The code implements and compares two CNN architectures: AlexNet and a ResNet-like model.

###Task 1: Implement AlexNet
1. Sequential Model Definition
  - The model is built using the Sequential() API in TensorFlow/Keras.
2. Convolutional and Pooling Layers
  - The first Conv2D layer (96 filters, 11×11 kernel, stride 4) extracts low-level features.
  - MaxPooling layers (3×3 pool size, stride 2) reduce spatial dimensions.
  - Additional Conv2D layers (256, 384, 384, 256 filters) extract deeper features.
  - Padding = 'same' ensures output dimensions remain the same after convolutions.
3. Fully Connected Layers and Dropout
  - The Flatten() layer converts 2D feature maps into a 1D vector.
  - Two Dense layers (4096 neurons) with ReLU activation for learning complex patterns.
  - Dropout (50%) prevents overfitting.
  - The final Dense layer (10 neurons, Softmax activation) outputs probabilities for 10 classes.
4. Model Summary
  - model.summary() prints the architecture details.


###Task 2: Implement a ResNet-like Model
1. Define a Residual Block
  - A residual block consists of two Conv2D layers with 64 filters.
  - A skip connection (Add()) adds the input to the output before activation, allowing deeper networks to train efficiently.
2. Build the ResNet Model
  - Initial Conv2D layer (64 filters, 7×7 kernel, stride 2) extracts initial features.
  - MaxPooling (3×3, stride 2) reduces spatial size.
  - Two residual blocks improve gradient flow and learning efficiency.
  - The model ends with Flatten, Dense (128 neurons, ReLU), and Softmax layers.
3. Model Summary
  - resnet.summary() prints the architecture details.









