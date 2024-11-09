#!/usr/bin/env python3

## Imports

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import sys
import os

from classification_models.tfkeras import Classifiers
from datetime import datetime
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Lambda
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# DATA ---------------------------------------------------------------

## Arguments

script_name = sys.argv[0]
batch_size = int(sys.argv[1])
dataset_name = sys.argv[2] if len(sys.argv) > 2 else "unknown"

start_time = datetime.now()
print(f"{start_time}: starting script {script_name}")


## Data linking

data_path = Path("./data/train")
test_path = Path("./data/test")
output_path = Path("./trained")
output_path.mkdir(exist_ok=True, parents=True)
(output_path / "models").mkdir(exist_ok=True)


## Loading images

df = pd.read_csv(data_path/"info.csv")
df["class"] = df["label"].astype(str)
df["filename"] = str(data_path) + "/" + df["class"] + "/" + df["img_name"] + ".png"

### Reduce huge classes
# Before oversampling
classes, counts = np.unique(df["class"], return_counts=True)
max_n_samples = 1000

print("classes", classes)
print("counts", counts)

df_sample = []
for i, cl in enumerate(classes):
    if counts[i] > max_n_samples:
        df_cl = df[df["class"]==cl].sample(max_n_samples)
        df_sample.append(df_cl)
    else:
        df_cl = df[df["class"]==cl]
        df_sample.append(df_cl)
df = pd.concat(df_sample).reset_index()

### Splitting
X = df[["subject", "filename"]]
y = df["class"]
groups = df["subject"]

gss = GroupShuffleSplit(n_splits=1, test_size=0.3)

train_index, test_index = next(gss.split(X, y, groups=groups))
df_train = df.iloc[train_index]
df_val = df.iloc[test_index]

df["split"] = "train"
df.loc[test_index, "split"] = "test"
df.to_csv(output_path / "dataset.csv")

### Compute class weights
class_weights = compute_class_weight('balanced',
                                   classes=np.unique(df_train["class"]),
                                   y=df_train["class"])
class_weights = dict(enumerate(class_weights))

print("\nCounts in training split:")
print(np.unique(df_train["class"], return_counts=True))
print("Counts in validation split:")
print(np.unique(df_val["class"], return_counts=True))


# GENERATOR ---------------------------------------------------------------

## Augmentation

train_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest',
    data_format="channels_last",
)

val_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
)


## Generator

image_size = (224, 224)

train_generator = train_datagen.flow_from_dataframe(
    df_train,
    x_col='filename',
    y_col='class',
    target_size=image_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    #seed=42,
)

val_generator = val_datagen.flow_from_dataframe(
    df_val,
    x_col='filename',
    y_col='class',
    target_size=image_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=True,
    #seed=42,
)

classes = np.unique(train_generator.labels)
num_classes = len(classes)


# TRAINING ---------------------------------------------------------------

## Model

model_name = 'resnet18'
ResNet18, preprocess_input = Classifiers.get(model_name)
base_model = ResNet18(
    input_shape=(image_size[0], image_size[1], 3),
    weights='imagenet',
    include_top=False,
)
base_model.trainable = True

### Convert input to RGB
def grayscale_to_rgb(x):
    return tf.image.grayscale_to_rgb(x)

input_layer = tf.keras.layers.Input(shape=(image_size[0], image_size[1], 1))
rgb_input = Lambda(grayscale_to_rgb)(input_layer)

### Compose together
# Add classification layers
x = base_model(rgb_input)
x = keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1000, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(x)
x = tf.keras.layers.Dropout(0.5)(x)
output = keras.layers.Dense(num_classes, activation='softmax')(x)
# Compose model
model = keras.Model(inputs=input_layer, outputs=output)

### Compilation
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())
print("\nNum GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


## Training

# Callbacks
checkpoints = keras.callbacks.ModelCheckpoint(
    output_path / "models/ep{epoch:03d}-loss{val_loss:.2f}.h5", save_best_only=True
)

early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.1,
    patience=10,
)

num_epochs = 250

history = model.fit(
    train_generator,
    epochs=num_epochs,
    callbacks=[checkpoints, early_stopping, reduce_lr],
    validation_data=val_generator,
    class_weight = class_weights,
    shuffle=True,
    verbose=2,
)

best_loss = np.min(history.history['val_loss'])
best_epoch = np.argmin(history.history['val_loss'])
best_acc = history.history['val_accuracy'][best_epoch]


# OUTPUTS ---------------------------------------------------------------

## Generate report

for property in ["accuracy", "loss"]:
    plt.clf()
    plt.plot(history.history[property])
    plt.plot(history.history[f"val_{property}"])
    plt.xlabel("epochs")
    plt.ylabel(property)
    plt.title(f"model {property} during training")
    plt.legend(["training", "validation"])
    plt.savefig(output_path / f"{property}.png")

# learning rate
plt.clf()
plt.plot(np.log10(history.history["lr"]))
plt.xlabel("epochs")
plt.ylabel("learning rate (log10)")
plt.title(f"learning rate during training")
plt.savefig(output_path / "lr.png")

# csv table
history_df = pd.DataFrame(history.history)
history_df.to_csv(output_path / "history.csv")

print("\nTop 5 epochs - val. accuracy:")
print(history_df.sort_values(by="val_accuracy").head(5))
print("\nTop 5 epochs - val. loss:")
print(history_df.sort_values(by="val_loss", ascending=True).head(5))

end_time = datetime.now()
print(f"\n{end_time}: finishing, writing report")

with open(output_path / "report.html", 'w') as f:
    # head
    f.writelines([
       '<!DOCTYPE html>',
       '  <head>',
       '    <meta charset="utf-8" />',
       f"    <title>Example: Report for {script_name}</title>"
       '  </head>'])
    #body
    f.writelines([
       '<body>',
       f"<h1>Report for {script_name}</h1>",
       f"<h2>Settings</h2>",
       "<ul>",
       f"<li>model: {model_name}</li>",
       f"<li>data: {dataset_name}</li>",
       f"<li>batch size: {batch_size}</li>",
       "</ul>"
       f"<h2>Model description</h2>",
       "<div>"])
    model.summary(print_fn=lambda x: f.write(x + '<br />'))
    f.writelines([
       "</div>",
       f"<h2>Results</h2>",
       "<ul>",
       f"<li>start time: {start_time}</li>",
       f"<li>end time: {end_time}</li>",
       f"<li>duration: {end_time-start_time}</li>",
       f"<li>best val. loss: {best_loss}</li>",
       f"<li>epoch of the best val. loss: {best_epoch}</li>",
       f"<li>val. accuracy in that epoch: {best_acc:.2f} %</li>",
       "</ul>"
       f"<h2>Plots</h2>",
       "<p><img src='./accuracy.png'></p>",
       "<p><img src='./loss.png'></p>",
       "<p><img src='./lr.png'></p>",
       '</body>',
    ])


# PREDICTION ---------------------------------------------------------------
    
print("\nPredictions -------------------------------------------------------")

best_model_name = [bmp for bmp in os.listdir(output_path/"models") if bmp.startswith(f"ep{best_epoch+1:03d}")][0]
best_model = tf.keras.models.load_model(output_path/"models"/best_model_name)
print(f"- model: {best_model_name}")

### Test data
df_test = pd.read_csv(test_path/"info.csv")
df_test["class"] = df_test["label"].astype(str)
df_test["filename"] = str(test_path) + "/" + df_test["class"] + "/" + df_test["img_name"] + ".png"

test_datagen = ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization=True,
    horizontal_flip=False,
    vertical_flip=False,
    data_format="channels_last",
)

### Test generator
test_generator = test_datagen.flow_from_dataframe(
    df_test,
    x_col='filename',
    y_col='class',
    target_size=image_size,
    color_mode='grayscale',
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False,
)

### Predict
pred_probs = best_model.predict(test_generator, batch_size=batch_size)
pred_label = np.argmax(pred_probs, axis=-1)

### Save
df_test["predicted_label"] = pred_label
df_test.to_csv(output_path/"test_dataset.csv")


# STATISTICS ---------------------------------------------------------------

print("\nPer slice metrics:")
accuracy = 100. * np.sum(pred_label == df_test["label"]) / len(pred_label)
print(f"- accuracy: {accuracy:.2f} %")
acc_tol = 100. * np.sum(np.abs(pred_label - df_test["label"]) <= 1) / len(pred_label)
print(f"- accuracy with +-1 grade tolerance: {acc_tol:.2f}")

conf_m = confusion_matrix(df_test["label"], pred_label)
print("- confusion matrix:")
print(conf_m)

plt.clf()
cm_plot = ConfusionMatrixDisplay(conf_m, display_labels=['0', '1', '2', '3A', '3B', '4'])
cm_plot.plot(cmap="Blues")
plt.title(f"predictions on test dataset")
plt.savefig(output_path / "conf_matrix_slices.png")


## Per patient

per_patient = []
pp_columns = ["subject", "right", "left", "pred_right", "pred_left"]
df_test["side"] = df_test["side"].replace({'R': 'right', 'L': 'left'})

for subject in df_test["subject"].unique():
    df_s = df_test[df_test["subject"] == subject]
    label_r = df_s[df_s["side"]=="right"]["label"].max()
    label_l = df_s[df_s["side"]=="left"]["label"].max()
    pred_r = df_s[df_s["side"]=="right"]["predicted_label"].max()
    pred_l = df_s[df_s["side"]=="left"]["predicted_label"].max()
    per_patient.append([subject, label_r, label_l, pred_r, pred_l])

df_pp = pd.DataFrame(per_patient, columns=pp_columns)
df_pp.to_csv(output_path/"maxed_scores.csv")

all_labels = np.concatenate((df_pp["right"], df_pp["left"]), axis=None)
all_preds = np.concatenate((df_pp["pred_right"], df_pp["pred_left"]), axis=None)

### Metrics
print("\nPer patient metrics")
pp_accuracy = 100. * np.sum(all_labels == all_preds) / len(all_labels)
print(f"- accuracy: {pp_accuracy:.2f} %")
pp_acc_tol = 100. * np.sum(np.abs(all_labels - all_preds) <= 1) / len(all_labels)
print(f"- accuracy with +-1 grade tolerance: {pp_acc_tol:.2f}")

conf_m = confusion_matrix(all_labels, all_preds)
print("- confusion matrix:")
print(conf_m)

plt.clf()
cm_plot = ConfusionMatrixDisplay(conf_m, display_labels=['0', '1', '2', '3A', '3B', '4'])
cm_plot.plot(cmap="Blues")
plt.title(f"predictions on test dataset")
plt.savefig(output_path / "conf_matrix.png")

with open("/mnt/personal/opltfili/knosp/outputs/model_comparison.csv", 'a') as f:
    f.write(f"{script_name},{batch_size},{end_time-start_time},{best_model_name},{accuracy},{acc_tol},{pp_accuracy},{pp_acc_tol}\n")

print("All done.")
