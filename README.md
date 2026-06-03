# Convolutional Neural Network (CNN)

This repository provides the user with an easy and simple way of designing a CNN in Python.
<br>
It allows the user to treat layers as blocks which can be configured and arranged as needed.


### Initialization
<hr>

The user should first create a Python list called `layer_configs` containing each of the layers in the desired order:
<br>

#### Example: LeNet-5

```
layer_configs = [
        ConvolutionalLayer(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        AveragePooling(kernel_size=2, stride=2),
        ConvolutionalLayer(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        AveragePooling(kernel_size=2, stride=2),
        ConvolutionalLayer(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0, padding_mode="reflect"),
        Flatten(),
        LinearLayer(in_features=120, out_features=84),
        Nonlinearity(name="relu"),
        LinearLayer(in_features=84, out_features=10),
]
```

The user can then initialize the model, an optimizer, and a criterion:

```
model = CNN(layer_configs=layer_configs).to(device)
criterion_config = Criterion(name="cross_entropy")
optimizer_config = Optimizer(name="AdamW", lr=1e-3, weight_decay=0.01)
```

### Training
<hr>

To train the model, the user should initialize a data loader with a batch size:
```
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```
Then call `model.fit()`

The `.fit()` method accepts `train_loader, val_loader, num_epochs, layer_configs, criterion_config, optimizer_config` and trains the model for the number of epochs specified by `num_epochs`.
<br>

It returns a history dict:
```
history = {
    "train_loss": [],
    "val_loss": [],
    "val_acc": []
}
```
The best validation loss across the epochs, and the epoch which achieved the best validation loss.
<br>
The history dict can be used to plot the training and validation loss as well as the validation accuracy after training. This can be useful for visualization and documentation, but can also be left out if desired.

```
history, best_val_loss, best_val_epoch = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        layer_configs=layer_configs,
        criterion_config=criterion_config,
        optimizer_config=optimizer_config,
        )
```

### Results
<hr>

`model.fit()` displays a log during training which looks like:
<br><br>

<div align="center">
  <img width="454" height="61" alt="best_validation_loss" src="https://github.com/user-attachments/assets/bb84cb07-2263-4009-8796-fa8911d0ee53" />
</div>

<br>
And as seen in the image above also shows the best validation loss and the epoch at which it took place.
<br>

The weights which resulted in the lowest validation loss are saved to `trained_weights.pth` during training.
<hr>

<div align="center">
  <strong>Thanks for reading!</strong>
</div>
