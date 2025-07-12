import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Try to import TensorFlow
try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import Model
    from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization, Activation, Conv1D,
        MaxPooling1D, UpSampling1D, concatenate
    )
    from tensorflow.keras.optimizers import Adam
    
    print(f"TensorFlow version: {tf.__version__}")
    
except ImportError as e:
    print(f"Error importing TensorFlow: {e}")
    print("Please install TensorFlow with: pip install tensorflow")
    raise

# 1. Loading geoelectric data
print("1. Loading geoelectric data...")
try:
    # Try to load from the local path first
    csv_path = "c:/Games/!Projects/Time-Domain Electromagnetics/geoelectric_models.csv"
    if not os.path.exists(csv_path):
        # Try relative path
        csv_path = "geoelectric_models.csv"
        if not os.path.exists(csv_path):
            # Fall back to Kaggle path if local file doesn't exist
            csv_path = "/kaggle/input/geoelectric-models/geoelectric_models.csv"
    
    df = pd.read_csv(csv_path)
    print(f"Successfully loaded data from {csv_path}")
except Exception as e:
    print(f"Error loading data: {e}")
    raise

# Extract features and targets
X_raw = df[[f'dBzdt_{i}' for i in range(1, 41)]].values
y_raw = df[[f'rho_{i}' for i in range(1, 21)] + [f'thk_{i}' for i in range(1, 20)]].values

print(f"Raw data shapes - X: {X_raw.shape}, y: {y_raw.shape}")
print(f"X range: [{X_raw.min():.2e}, {X_raw.max():.2e}], y range: [{y_raw.min():.2f}, {y_raw.max():.2f}]")

# Check for non-positive values in y_raw
if np.any(y_raw <= 0):
    print(f"Warning: y_raw contains {np.sum(y_raw <= 0)} non-positive values.")
    # Add a small offset to avoid log(0) issues
    y_raw = y_raw + 1e-10

# Apply log transform with clipping to avoid extreme values
X_raw_clipped = np.clip(X_raw, 1e-10, None)  # Clip to avoid very small values
y_raw_clipped = np.clip(y_raw, 1e-10, None)  # Clip to avoid very small values

X = np.log10(X_raw_clipped)
y = np.log10(y_raw_clipped)

# Check for any remaining NaN or inf values
if np.any(np.isnan(X)) or np.any(np.isinf(X)):
    print("Warning: X contains NaN or inf values after log transform. Fixing...")
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    print("Warning: y contains NaN or inf values after log transform. Fixing...")
    y = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)

# Split data with stratification if possible
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

# Create validation set from training data
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, shuffle=True
)

print(f"Train/Val/Test split: {X_train.shape[0]}/{X_val.shape[0]}/{X_test.shape[0]} samples")

# Use StandardScaler for normalization
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_val = scaler_y.transform(y_val)
y_test = scaler_y.transform(y_test)

# Reshape for CNN input
X_train = X_train[..., np.newaxis]  # Shape: (num_samples, 40, 1)
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

y_train = y_train[..., np.newaxis]  # Shape: (num_samples, 39, 1)
y_val = y_val[..., np.newaxis]
y_test = y_test[..., np.newaxis]

print(f"Final data shapes:")
print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
print("Input features: dBzdt_1 to dBzdt_40")
print("Output features: rho_1 to rho_20 + thk_1 to thk_19")

# 2. Define a simple model
def SimpleGPRNet(im_height=40, neurons=8, kern_sz=3):
    """
    A simplified version of the GPRNet model with fewer parameters
    """
    input_img = Input((im_height, 1))
    
    # Encoder
    x = Conv1D(neurons, kernel_size=kern_sz, activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    pool1 = MaxPooling1D(2, padding='same')(x)  # 20
    
    x = Conv1D(neurons*2, kernel_size=kern_sz, activation='relu', padding='same')(pool1)
    x = BatchNormalization()(x)
    pool2 = MaxPooling1D(2, padding='same')(x)  # 10
    
    x = Conv1D(neurons*4, kernel_size=kern_sz, activation='relu', padding='same')(pool2)
    x = BatchNormalization()(x)
    encoded = MaxPooling1D(2, padding='same')(x)  # 5
    
    # Decoder
    x = Conv1D(neurons*4, kernel_size=kern_sz, activation='relu', padding='same')(encoded)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  # 10
    
    x = Conv1D(neurons*2, kernel_size=kern_sz, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  # 20
    
    x = Conv1D(neurons, kernel_size=kern_sz, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = UpSampling1D(2)(x)  # 40
    
    # Adjust output length to 39
    x = Conv1D(neurons, kernel_size=2, padding='valid')(x)  # 40 - 2 + 1 = 39
    output = Conv1D(1, kernel_size=1, activation='linear')(x)
    
    model = Model(inputs=input_img, outputs=output)
    return model

# Define custom metrics for evaluation
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# 3. Model Training and Evaluation
# Define hyperparameters
EPOCHS = 100
BATCH_SIZE = 32
PATIENCE = 20
MODEL_DIR = "models"

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# Define model checkpoint callback
checkpoint_path = os.path.join(MODEL_DIR, "best_model.h5")
model_checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_loss',
    save_best_only=True,
    save_weights_only=False,
    mode='min',
    verbose=1
)

# Define callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Create and compile the model
print("\n3. Creating and training the model...")
model = SimpleGPRNet(
    im_height=40,
    neurons=8,
    kern_sz=3
)

model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),  # Use gradient clipping
    loss='mse',
    metrics=[rmse, r2]
)

# Print model summary
model.summary()

# Train the model with a conservative approach
print("\nTraining the model...")
try:
    # Start with a very small batch size
    initial_epochs = 5
    print(f"Initial training with batch size {BATCH_SIZE//4}...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE//4,
        epochs=initial_epochs,
        verbose=1,
        validation_data=(X_val, y_val)
    )
    
    # Continue with full training
    print(f"Continuing training with batch size {BATCH_SIZE}...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[
            model_checkpoint,
            reduce_lr,
            early_stopping
        ]
    )
except Exception as e:
    print(f"Error during training: {e}")
    raise

# Evaluate the model
print("\nEvaluating the model...")
try:
    # Try to load the best model if it was saved
    if os.path.exists(checkpoint_path):
        print("Loading the best model...")
        best_model = tf.keras.models.load_model(
            checkpoint_path, 
            custom_objects={'rmse': rmse, 'r2': r2}
        )
    else:
        print("Best model not saved, using current model...")
        best_model = model
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = best_model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Loss (MSE): {test_results[0]:.6f}")
    print(f"Test RMSE: {test_results[1]:.6f}")
    print(f"Test R²: {test_results[2]:.6f}")
    
    # Make predictions
    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Calculate additional metrics
    train_mse = mean_squared_error(y_train.flatten(), y_train_pred.flatten())
    val_mse = mean_squared_error(y_val.flatten(), y_val_pred.flatten())
    test_mse = mean_squared_error(y_test.flatten(), y_test_pred.flatten())
    
    train_mae = mean_absolute_error(y_train.flatten(), y_train_pred.flatten())
    val_mae = mean_absolute_error(y_val.flatten(), y_val_pred.flatten())
    test_mae = mean_absolute_error(y_test.flatten(), y_test_pred.flatten())
    
    train_r2 = r2_score(y_train.flatten(), y_train_pred.flatten())
    val_r2 = r2_score(y_val.flatten(), y_val_pred.flatten())
    test_r2 = r2_score(y_test.flatten(), y_test_pred.flatten())
    
    print("\nDetailed Metrics:")
    print(f"Train - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}, R²: {train_r2:.6f}")
    print(f"Val   - MSE: {val_mse:.6f}, MAE: {val_mae:.6f}, R²: {val_r2:.6f}")
    print(f"Test  - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}, R²: {test_r2:.6f}")
    
    # 4. Visualization
    print("\n4. Visualizing results...")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['r2'], label='Train R²')
    plt.plot(history.history['val_r2'], label='Validation R²')
    plt.title('Training and Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'training_history.png'))
    plt.show()
    
    # Plot predictions vs actual values
    plt.figure(figsize=(15, 5))
    
    # Training data
    plt.subplot(1, 3, 1)
    plt.scatter(y_train.flatten(), y_train_pred.flatten(), alpha=0.1, color='blue')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
    plt.title(f'Training Data (R² = {train_r2:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    # Validation data
    plt.subplot(1, 3, 2)
    plt.scatter(y_val.flatten(), y_val_pred.flatten(), alpha=0.1, color='green')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--')
    plt.title(f'Validation Data (R² = {val_r2:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    # Test data
    plt.subplot(1, 3, 3)
    plt.scatter(y_test.flatten(), y_test_pred.flatten(), alpha=0.1, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title(f'Test Data (R² = {test_r2:.4f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, 'prediction_scatter.png'))
    plt.show()
    
except Exception as e:
    print(f"Error during evaluation: {e}")
    raise

print("\nTraining and evaluation complete. Model and visualizations saved to:", MODEL_DIR)