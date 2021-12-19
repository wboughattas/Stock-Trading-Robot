from Models.import_batches import import_batches
import sklearn
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms


def initialize_ci():
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

    # Run data through ConvolutionalNeuralNetwork
    # train_cnn(X_train, y_train)

    # Test accuracy of ConvolutionalNeuralNetwork
    # test_cnn(X_test, y_test)

    # Run data through DecisionTreeClassifier
    max_depths = [1, 2, 5, 10, 20, 50]
    tree_estimators = train_trees(X_train, y_train, max_depths)

    # Test accuracy of DecisionTreeClassifier
    test_trees(X_test, y_test, tree_estimators)


def train_cnn(X_train, y_train):
    # Convert X_train into a tensor normalized between [-1, 1]
    X_tensor = X_to_normalized_tensor(X_train.reshape(-1, 3, 32, 32))
    y_tensor = torch.tensor(y_train, dtype=torch.int64)

    # Create CNN model
    model = torch.nn.Sequential(

        # Channels from 3 to 16
        torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),

        # Second Pass at 16 Channels
        torch.nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(16),
        torch.nn.ReLU(),

        # Image Size from 32x32 to 16x16
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # Channels from 16 to 32
        torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),

        # Second Pass at 32 Channels
        torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(32),
        torch.nn.ReLU(),

        # Image Size from 16x16 to 8x8
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # Channels from 32 to 64
        torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),

        # Second Pass at 64 Channels
        torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(64),
        torch.nn.ReLU(),

        # Image Size from 8x8 to 4x4
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # Channels from 64 to 128
        torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),

        # Second Pass at 128 Channels
        torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(128),
        torch.nn.ReLU(),

        # Image Size from 4x4 to 2x2
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # Channels from 128 to 256
        torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),

        # Second Pass at 256 Channels
        torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        torch.nn.BatchNorm2d(256),
        torch.nn.ReLU(),

        # Image Size from 2x2 to 1x1
        torch.nn.MaxPool2d(kernel_size=2, stride=2),

        # Final Linear Transformation to Output
        torch.nn.Flatten(),
        torch.nn.Linear(in_features=256, out_features=10)
    )

    num_epoch = 10
    batch_size = 500

    # Train model
    loss = torch.nn.CrossEntropyLoss()
    # Initial learning rate set to 0.05
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # Divide learning rate by 2 after every batch
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=9, verbose=True)

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
    torch.save(model, 'cnn_trained.pkl')


def test_cnn(X_test, y_test):
    # Convert X_test into a tensor normalized between [-1, 1]
    X_tensor = X_to_normalized_tensor(X_test.reshape(-1, 3, 32, 32))
    y_tensor = torch.tensor(y_test, dtype=torch.int64)

    # Load saved model
    model = torch.load('cnn_trained.pkl')

    # Keep track of correct vs total predictions for accuracy
    c_predict = 0
    t_predict = 0

    # Run test data through model.
    y_pred = model(X_tensor)

    # Gets the maximum value of the predicted set along dim = 1 (row-wise) where the index of the highest prediction is
    # the corresponding class.
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
    for estimator in tree_estimators:
        score = 100 * estimator.score(X_test, y_test)
        print('Score of DecisionTreeClassifier with max_depth =  %d: %.3f%%' % (estimator.max_depth, score))

        # Plot and export tree image to file
        plt.figure()
        sklearn.tree.plot_tree(estimator)
        plt.savefig('DecisionTreeClassifier_MaxDepth_%d.png' % estimator.max_depth, format='png')


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
