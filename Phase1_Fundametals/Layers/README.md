# Base blocks for a neural network
The torch.nn module in PyTorch provides various building blocks for constructing and training neural networks. Here are the main building blocks:

## layers
- Linear layers
    - nn.Linear -> Fully connected Dense layer

- Convolutional layers
    - nn.Conv1d -> This is for time series
    - nn.Conv2d -> This is for images
    - nn.Conv3d -> This is for volumetric data

- Recurrent layers
    - nn.RNN -> Recurrent neural network
    - nn.LSTM -> Long-Short memory
    - nn.GRU -> Gated Recurrent Unit

- Normalization layers
    - nn.BatchNorm1d -> Batch normalization for 1D
    - nn.BatchNorm2d -> Batch normalization for 2d
    - nn.LayerNorm -> Normalize across the last dimension
    - nn.GroupNorm -> NOrmalize for a group of channels

- Pooling layers
    - nn.MaxPool2d -> Maxpooling for downsampling
    - nn.AvgPool2d -> Average for downsampling
    - nn.AdaptiveAvgPool2d -> Fixed sized feature map

- Dropout
    - nn.Dropout -> Applies random zeros in the training to avoid overfitting

## Activation functions
- nn.ReLU() -> its used in the dense layers
- nn.Sigmoid() -> Its use for binary classification and if used at the end give back probabilities
- nn.Tanh() -> Its used for classification and the output its between -1 and 1
- nn.Softmax() -> Convert logits in to probabilities, its used for multi task
- nn.LeakyReLu() -> Helps imporve the gradient flow, and avoid dead neurons buts add another hyperparameter
- nn.ELU -> Helps improve the gradient flow, and avoid dead neurons but its more computationally expensive

## Loss functions
- nn.MSELoss() -> Regression tasks where the goal is to minimize the squared difference between predicted and true values. sensitive to large erros but the outliers may impact too much
- nn.CrossEntropyLoss() -> Multiclass classification task, 
- nn.BCELoss() -> Binary classification tasks, works good for binary outcomes, the output must be between o and 1, often used with Sigmoid
- nn.BCEWithLogitsLoss() -> Binary classification, combines Sigmoid activation and nn.BCELoss for numerical stability.
- nn.L1Loss() -> Regression tasks where you want to minimize the absolute differences. More robust to outliers
- nn.NLLLoss() -> Works with probabilities from nn.LogSoftmax for classification. Log probabilities
- nn.HingeEmbeddingLoss() -> Semi-supervised learning and similarity learning. Helps in learning representations where distances matter.

## Containers
- nn.Sequential() -> Stack layers sequential
- nn.Module() -> Base class for all Neural Network
- nn.ModuleList() -> List of modules in a list
- nn.ModuleDict() -> Submodules in a dict

## Utilities
- nn.Flatten() -> Flattens the input of the tensor