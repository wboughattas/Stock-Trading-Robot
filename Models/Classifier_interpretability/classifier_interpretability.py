from Models.import_batches import *
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from collections import OrderedDict


def initialize_ci():
    torch.set_printoptions(sci_mode=False)

    # Get and sort data from dataset into respective containers
    dataset = import_batches()
    metadata = dataset[0]
    train_data = dataset[1:6]
    test_data = dataset[6]

    # Combine batches into training data
    X_train = np.empty(shape=(0, 3072), dtype=np.uint8)
    y_train = np.empty(shape=(0,), dtype=np.int32)

    for batch in train_data:
        X_train = np.append(X_train, batch[b'data'], axis=0)
        y_train = np.append(y_train, batch[b'labels'])

    # Turn test_batch into testing data
    X_test = np.array(test_data[b'data'], dtype=np.uint8)
    y_test = np.array(test_data[b'labels'], dtype=np.int32)

    for epochs in [5, 10, 20]:

        # Run data through ConvolutionalNeuralNetwork
        # train_cnn(X_train, y_train, epochs)

        # Test accuracy of ConvolutionalNeuralNetwork
        # test_cnn(X_test, y_test, epochs)

        # Activation Maximization
        for id in range(10):
            activation_maximization(id, epochs)

    # Run data through DecisionTreeClassifier
    max_depths = [1, 2, 5, 10, 20, 50]


# tree_estimators = train_trees(X_train, y_train, max_depths)

# Test accuracy of DecisionTreeClassifier
# test_trees(X_test, y_test, tree_estimators)


def train_cnn(X_train, y_train, epochs):
    # Convert X_train into a tensor normalized between [-1, 1]
    X_tensor = X_to_normalized_tensor(X_train.reshape(-1, 3, 32, 32))
    y_tensor = torch.tensor(y_train, dtype=torch.int64)

    # Create CNN model
    model = torch.nn.Sequential(OrderedDict([

        # Channels from 3 to 16
        ('conv_3_16', torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)),
        ('batch_16_1', torch.nn.BatchNorm2d(16)),
        ('activ_1', torch.nn.ReLU()),

        # Second Pass at 16 Channels
        ('conv_16_16', torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)),
        ('batch_16_2', torch.nn.BatchNorm2d(16)),
        ('activ_2', torch.nn.ReLU()),

        # Image Size from 32x32 to 16x16
        ('pool_1', torch.nn.MaxPool2d(kernel_size=2, stride=2)),

        # Channels from 16 to 32
        ('conv_16_32', torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)),
        ('batch_32_1', torch.nn.BatchNorm2d(32)),
        ('activ_3', torch.nn.ReLU()),

        # Second Pass at 32 Channels
        ('conv_32_32', torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)),
        ('batch_32_2', torch.nn.BatchNorm2d(32)),
        ('activ_4', torch.nn.ReLU()),

        # Image Size from 16x16 to 8x8
        ('pool_2', torch.nn.MaxPool2d(kernel_size=2, stride=2)),

        # Channels from 32 to 64
        ('conv_32_64', torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)),
        ('batch_64_1', torch.nn.BatchNorm2d(64)),
        ('activ_5', torch.nn.ReLU()),

        # Second Pass at 64 Channels
        ('conv_64_64', torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)),
        ('batch_64_2', torch.nn.BatchNorm2d(64)),
        ('activ_6', torch.nn.ReLU()),

        # Image Size from 8x8 to 4x4
        ('pool_3', torch.nn.MaxPool2d(kernel_size=2, stride=2)),

        # Channels from 64 to 128
        ('conv_64_128', torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)),
        ('batch_128_1', torch.nn.BatchNorm2d(128)),
        ('activ_7', torch.nn.ReLU()),

        # Second Pass at 128 Channels
        ('conv_128_128', torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)),
        ('batch_128_2', torch.nn.BatchNorm2d(128)),
        ('activ_8', torch.nn.ReLU()),

        # Image Size from 4x4 to 2x2
        ('pool_4', torch.nn.MaxPool2d(kernel_size=2, stride=2)),

        # Channels from 128 to 256
        ('conv_128_256', torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)),
        ('batch_256_1', torch.nn.BatchNorm2d(256)),
        ('activ_9', torch.nn.ReLU()),

        # Second Pass at 256 Channels
        ('conv_256_256', torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
        ('batch_256_2', torch.nn.BatchNorm2d(256)),
        ('activ_10', torch.nn.ReLU()),

        # Image Size from 2x2 to 1x1
        ('pool_5', torch.nn.MaxPool2d(kernel_size=2, stride=2)),

        # Final Linear Transformation to Output
        ('flat_1', torch.nn.Flatten()),
        ('lin_1', torch.nn.Linear(in_features=256, out_features=10))
    ]))

    num_epoch = epochs
    batch_size = 500

    # Train model
    loss = torch.nn.CrossEntropyLoss()
    # Initial learning rate set to 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # Divide learning rate by 2 after every batch
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5)

    for epoch in range(1, num_epoch + 1):

        for i in range(0, len(X_tensor), batch_size):
            X = X_tensor[i: i + batch_size]
            y = y_tensor[i: i + batch_size]

            y_pred = model(X)
            loss_value = loss(y_pred, y)

            model.zero_grad()
            loss_value.backward()
            optimizer.step()

        print('Epoch %d final minibatch had loss of %.4f' % (epoch, loss_value.item()))
        scheduler.step()

    # Output Trained model to file
    torch.save(model, 'cnn_trained_e%d.pkl' % num_epoch)


def test_cnn(X_test, y_test, epochs):
    # Convert X_test into a tensor normalized between [-1, 1]
    X_tensor = X_to_normalized_tensor(X_test.reshape(-1, 3, 32, 32))
    y_tensor = torch.tensor(y_test, dtype=torch.int64)

    # Load saved model
    model = torch.load('cnn_trained_e%d.pkl' % epochs)

    # Keep track of correct vs total predictions for accuracy
    c_predict = 0
    t_predict = 0

    # Run test data through model.
    y_pred = model(X_tensor)

    # Gets the maximum value of the predicted set along dim = 1 (row-wise) where the index of the highest prediction is the corresponding class.
    val, predictions = torch.max(y_pred, 1)

    # Count total predictions
    t_predict = y_tensor.size(0)

    # Determine correct predictions
    c_predict = (predictions == y_tensor).sum().item()

    accuracy = 100 * c_predict / t_predict

    print('Accuracy of CNN on Test Set: %.2f%%' % accuracy)


def train_trees(X_train, y_train, max_depths):
    tree_estimators = []

    for depth in max_depths:
        print('Training DecisionTreeClassifier with max_depth = %d' % depth)
        tree = sklearn.tree.DecisionTreeClassifier(max_depth=depth, random_state=0)
        tree.fit(X_train, y_train)
        tree_estimators.append(tree)

    return tree_estimators


def test_trees(X_test, y_test, tree_estimators):
    scores = []
    for estimator in tree_estimators:
        score = 100 * estimator.score(X_test, y_test)
        print('Score of DecisionTreeClassifier with max_depth =  %d: %.3f%%' % (estimator.max_depth, score))

        # Plot and export tree image to file
        plt.figure()
        sklearn.tree.plot_tree(estimator, max_depth=2)
        plt.savefig('DecisionTreeClassfier_MaxDepth_%d.png' % estimator.max_depth, format='png')


def X_to_normalized_tensor(data):
    # Convert ndarray into tensor, with reshaped values to split 3072 into (3, 32, 32) and dtype = torch.float32
    tensor = torch.tensor(data, dtype=torch.float32)

    # Rescale tensor to range [0, 1]
    tensor = tensor / 255

    # Normalize tensor within range [-1, 1]
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]

    return tensor


def normalize_tensor(tensor):
    mean = torch.tensor([0.5])
    std = torch.tensor([0.5])

    tensor = (tensor - mean) / std

    return tensor


def denormalize_tensor(tensor):
    mean = torch.tensor([0.5, 0.5, 0.5])
    std = torch.tensor([0.5, 0.5, 0.5])

    tensor = (tensor * std[:, None, None]) + mean[:, None, None]

    return tensor


def activation_maximization(id, epochs):
    # Create initial image and convert to tensor
    base_img = np.zeros([32, 32, 3], dtype=np.uint8)
    for i in range(3):
        base_img[:, :, i] = np.zeros([32, 32]) + 127

    x = X_to_normalized_tensor(base_img.reshape(-1, 3, 32, 32))
    x.requires_grad = True

    # Define Classes
    classes = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }

    # Load Model
    model = torch.load('cnn_trained_e%d.pkl' % epochs)

    # Set model to evaluation only (not training)
    model.eval()

    num_epochs = 5000
    display = 5000
    class_id = id
    lr = 0.5

    for epoch in range(1, num_epochs + 1):

        # Get class probability predictions from the model
        y_pred = model(x)

        # Compute Softmax and take value at target class
        soft = torch.softmax(y_pred, dim=1)
        y = soft[0][class_id]

        # Backpropagate with respect to x to populate x.grad
        y.backward()

        # Step x towards desired value
        x.data += lr * x.grad

        # Reset x.grad for next iteration
        x.grad.data.zero_()

        # Output every display iterations (display = num_epochs for only final image)
        if epoch % display == 0:
            visualize_image(x, 'cnn%d_class_ %s_epoch_%d ' % (epochs, classes[class_id], epoch))


def visualize_image(image_tensor, title):
    # print(image_tensor)

    # Convert Tensor to Image, Denormalize and Rescale
    image_copy = image_tensor.cpu()
    image_copy = denormalize_tensor(image_copy.clone().detach()).numpy()[0]
    image_copy = (image_copy.transpose(1, 2, 0) * 255).astype(int)
    image_copy = image_copy.clip(0, 255)

    # Plot Image
    plt.figure()
    plt.imshow(image_copy)
    plt.title(title)
    plt.savefig('./out_img/' + title)
