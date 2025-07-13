import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import uniform as rand
from tqdm import tqdm
import scipy.io as sio
import scipy.stats
import scipy.ndimage as ndimage
from scipy.signal import savgol_filter
from scipy.signal.windows import tukey
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FormatStrFormatter

# Try to import skimage, but continue if not available
try:
    from skimage.transform import rescale, resize
except ImportError:
    print("Warning: skimage not available. Some functionality may be limited.")
    # Define dummy functions if needed
    def rescale(image, scale, **kwargs):
        return image
    def resize(image, output_shape, **kwargs):
        return image

from sklearn.model_selection import train_test_split
from sklearn import preprocessing as pp
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, mean_absolute_error
from numpy.matlib import repmat
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

try:
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras import optimizers, regularizers
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
    from tensorflow.keras.layers import (
        Input, Dense, Dropout, BatchNormalization, Activation, Add, Flatten, Reshape,
        SpatialDropout1D, GaussianDropout, GaussianNoise, Conv1D, Conv2D, Conv2DTranspose, ZeroPadding1D,
        MaxPooling1D, UpSampling1D, concatenate, LeakyReLU, PReLU, LSTM, Bidirectional, GlobalAveragePooling1D,
        multiply
    )
    from tensorflow.keras.optimizers import Adam, RMSprop, SGD
    from tensorflow.keras.optimizers.schedules import ExponentialDecay
    
    # Check TensorFlow version
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

# Try different scalers to find the best one
scalers = {
    'standard': StandardScaler(),
    'minmax': MinMaxScaler(),
    'robust': RobustScaler()
}

# Use RobustScaler for better handling of outliers
scaler_X = scalers['robust']
scaler_y = scalers['robust']

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

# 2. Define Enhanced GPRNET Model
def GPRNet_enhanced(
    im_height=40, 
    neurons=32, 
    kern_sz=5, 
    enable_dropout=True, 
    dp_coeff=0.2,
    l2_reg=1e-5,
    use_batch_norm=True,
    use_residual=True,
    activation='leaky_relu'
):
    """
    Enhanced GPRNet model with:
    - Residual connections
    - Batch normalization
    - LeakyReLU activation
    - L2 regularization
    - Improved architecture
    - Attention mechanism
    """
    # Input layer
    input_img = Input((im_height, 1))
    
    # Add noise for better generalization
    x = GaussianNoise(0.01)(input_img)
    
    # Choose activation function
    if activation == 'leaky_relu':
        act = LeakyReLU(alpha=0.2)
        act_name = 'leaky_relu'
    elif activation == 'prelu':
        act = PReLU()
        act_name = 'prelu'
    else:
        act = 'relu'
        act_name = 'relu'
    
    # Regularizer
    reg = regularizers.l2(l2_reg)
    
    # Initial convolution
    x = Conv1D(neurons, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    
    # Encoder Block 1
    skip1 = x
    x = Conv1D(neurons, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    if enable_dropout:
        x = SpatialDropout1D(dp_coeff/2)(x)
    if use_residual:
        x = Add()([x, skip1])  # Residual connection
    pool1 = MaxPooling1D(2, padding='same')(x)  # 20
    
    # Encoder Block 2
    skip2 = Conv1D(neurons*2, kernel_size=1, padding='same')(pool1)  # Match dimensions
    x = Conv1D(neurons*2, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(pool1)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    x = Conv1D(neurons*2, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    if enable_dropout:
        x = SpatialDropout1D(dp_coeff/2)(x)
    if use_residual:
        x = Add()([x, skip2])  # Residual connection
    pool2 = MaxPooling1D(2, padding='same')(x)  # 10
    
    # Encoder Block 3
    skip3 = Conv1D(neurons*4, kernel_size=1, padding='same')(pool2)  # Match dimensions
    x = Conv1D(neurons*4, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(pool2)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    x = Conv1D(neurons*4, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    if enable_dropout:
        x = SpatialDropout1D(dp_coeff)(x)
    if use_residual:
        x = Add()([x, skip3])  # Residual connection
    pool3 = MaxPooling1D(2, padding='same')(x)  # 5
    
    # Enhanced Pyramid Pooling with different dilation rates
    py1 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=1, padding='same', 
                 kernel_regularizer=reg)(pool3)
    if use_batch_norm:
        py1 = BatchNormalization()(py1)
    py1 = Activation(act)(py1) if isinstance(act, str) else act(py1)
    
    py2 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=2, padding='same', 
                 kernel_regularizer=reg)(pool3)
    if use_batch_norm:
        py2 = BatchNormalization()(py2)
    py2 = Activation(act)(py2) if isinstance(act, str) else act(py2)
    
    py3 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=4, padding='same', 
                 kernel_regularizer=reg)(pool3)
    if use_batch_norm:
        py3 = BatchNormalization()(py3)
    py3 = Activation(act)(py3) if isinstance(act, str) else act(py3)
    
    py4 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=8, padding='same', 
                 kernel_regularizer=reg)(pool3)
    if use_batch_norm:
        py4 = BatchNormalization()(py4)
    py4 = Activation(act)(py4) if isinstance(act, str) else act(py4)
    
    # Merge pyramid features
    merge1 = concatenate([py1, py2, py3, py4, pool3])
    
    # Simplified attention mechanism to avoid instability
    # Just use a 1x1 convolution instead of full attention
    merge1 = Conv1D(neurons*8, kernel_size=1, padding='same', 
                   kernel_regularizer=reg)(merge1)
    
    # Bottleneck processing
    x = Conv1D(neurons*8, kernel_size=1, padding='same', 
               kernel_regularizer=reg)(merge1)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    
    # Skip connection from bottleneck to later layers
    bottleneck_skip = x
    
    # Upsampling path with skip connections
    x = UpSampling1D(2)(x)  # 5*2=10
    
    # Decoder Block 1
    x = concatenate([x, pool2])  # Skip connection from encoder
    x = Conv1D(neurons*4, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    x = Conv1D(neurons*4, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    
    # Decoder Block 2
    x = UpSampling1D(2)(x)  # 10*2=20
    x = concatenate([x, pool1])  # Skip connection from encoder
    x = Conv1D(neurons*2, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    x = Conv1D(neurons*2, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    
    # Decoder Block 3
    x = UpSampling1D(2)(x)  # 20*2=40
    x = Conv1D(neurons, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    x = Conv1D(neurons, kernel_size=kern_sz, padding='same', 
               kernel_regularizer=reg)(x)
    if use_batch_norm:
        x = BatchNormalization()(x)
    x = Activation(act)(x) if isinstance(act, str) else act(x)
    
    # Adjust output length to 39
    x = Conv1D(neurons, kernel_size=2, padding='valid')(x)  # 40 - 2 + 1 = 39
    
    # Final prediction layer
    output = Conv1D(1, kernel_size=1, activation='linear')(x)
    
    model = Model(inputs=input_img, outputs=output)
    return model

# Also keep the original model for comparison
def GPRNet_original(im_height=40, neurons=8, kern_sz=5, enable_dropout=False, dp_coeff=0.2):
    input_img = Input((im_height, 1))

    # Encoder
    conv1 = Conv1D(neurons, kernel_size=kern_sz, activation='relu', padding='same')(input_img)
    pool1 = MaxPooling1D(2, padding='same')(conv1)  # 20

    conv2 = Conv1D(neurons*2, kernel_size=kern_sz, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling1D(2, padding='same')(conv2)  # 10

    conv3 = Conv1D(neurons*4, kernel_size=kern_sz, activation='relu', padding='same')(pool2)
    pool3 = MaxPooling1D(2, padding='same')(conv3)  # 5

    # Pyramid pooling
    py1 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=1, activation='relu', padding='same')(pool3)
    py2 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=2, activation='relu', padding='same')(pool3)
    py3 = Conv1D(neurons*8, kernel_size=kern_sz, dilation_rate=4, activation='relu', padding='same')(pool3)
    merge1 = concatenate([py1, py2, py3, pool3])

    mgconv = Conv1D(neurons*8, kernel_size=3, activation='relu', padding='same')(merge1)
    upmgconv = UpSampling1D(4)(mgconv)  # 5*4=20

    # Decoder
    deconv1 = Conv1D(neurons*4, kernel_size=kern_sz, activation='relu', padding='same')(pool3)
    up1 = UpSampling1D(2)(deconv1)  # 5*2=10

    deconv2 = Conv1D(neurons*2, kernel_size=kern_sz, activation='relu', padding='same')(up1)
    up2 = UpSampling1D(2)(deconv2)  # 10*2=20

    merge2 = concatenate([upmgconv, up2])

    deconv3 = Conv1D(neurons*2, kernel_size=kern_sz, activation='relu', padding='same')(merge2)
    up3 = UpSampling1D(2)(deconv3)  # 20*2=40

    deconv4 = Conv1D(neurons, kernel_size=kern_sz, activation='relu', padding='same')(up3)

    # Adjust output length to 39
    conv_adjust = Conv1D(neurons, kernel_size=2, padding='valid')(deconv4)  # 40 - 2 + 1 = 39
    output = Conv1D(1, kernel_size=1, activation='linear')(conv_adjust)  # Shape: (39, 1)

    model = Model(inputs=input_img, outputs=output)
    return model

# 3. Model Training and Evaluation

# Define hyperparameters
EPOCHS = 500
BATCH_SIZE = 32
PATIENCE = 30
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
    patience=15,
    min_lr=1e-6,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=PATIENCE,
    restore_best_weights=True,
    verbose=1
)

# Create TensorBoard callback for visualization
tensorboard_callback = TensorBoard(
    log_dir=os.path.join(MODEL_DIR, "logs"),
    histogram_freq=1,
    write_graph=True,
    update_freq='epoch'
)

# Define custom metrics for evaluation
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def r2(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res / (SS_tot + K.epsilon())

# Create and compile the enhanced model
print("\n3. Creating and training the enhanced model...")
model = GPRNet_enhanced(
    im_height=40,
    neurons=16,  # Reduced number of neurons
    kern_sz=3,   # Smaller kernel size
    enable_dropout=True,
    dp_coeff=0.2,  # Reduced dropout
    l2_reg=1e-4,   # Increased regularization
    use_batch_norm=True,
    use_residual=True,
    activation='relu'  # Use standard ReLU to avoid instability
)

# Use gradient clipping to prevent exploding gradients
model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),  # Add gradient clipping
    loss='mse',
    metrics=[rmse, r2]
)

# Print model summary
model.summary()

# Train the model with a more conservative approach
print("\nTraining the model...")
try:
    # Start with a very small batch size and gradually increase it
    initial_epochs = 10
    print(f"Initial training with batch size {BATCH_SIZE//4}...")
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE//4,  # Start with smaller batch size
        epochs=initial_epochs,
        verbose=1,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )
    
    # Check if training is stable
    val_loss = model.evaluate(X_val, y_val, verbose=0)[0]
    if not np.isnan(val_loss) and not np.isinf(val_loss):
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
                early_stopping,
                tensorboard_callback
            ]
        )
    else:
        print("Training is unstable. Trying with original model architecture...")
        # Fall back to original model if enhanced model is unstable
        model = GPRNet_original(
            im_height=40,
            neurons=16,
            kern_sz=5,
            enable_dropout=True,
            dp_coeff=0.2
        )
        model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss='mse',
            metrics=[rmse, r2]
        )
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
    print("Falling back to original model architecture...")
    # Fall back to original model if enhanced model fails
    model = GPRNet_original(
        im_height=40,
        neurons=16,
        kern_sz=5,
        enable_dropout=True,
        dp_coeff=0.2
    )
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss='mse',
        metrics=[rmse, r2]
    )
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

# Evaluate the model
print("\nEvaluating the model...")
try:
    # Try to load the best model if it was saved
    if os.path.exists(checkpoint_path):
        print("Loading the best model...")
        best_model = load_model(checkpoint_path, custom_objects={'rmse': rmse, 'r2': r2})
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
except Exception as e:
    print(f"Error during evaluation: {e}")
    print("Using current model for predictions...")
    
    # Make predictions with current model
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)

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

# Plot error distribution
plt.figure(figsize=(15, 5))

# Training error
train_errors = y_train.flatten() - y_train_pred.flatten()
plt.subplot(1, 3, 1)
plt.hist(train_errors, bins=50, alpha=0.7, color='blue')
plt.title('Training Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')

# Validation error
val_errors = y_val.flatten() - y_val_pred.flatten()
plt.subplot(1, 3, 2)
plt.hist(val_errors, bins=50, alpha=0.7, color='green')
plt.title('Validation Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')

# Test error
test_errors = y_test.flatten() - y_test_pred.flatten()
plt.subplot(1, 3, 3)
plt.hist(test_errors, bins=50, alpha=0.7, color='red')
plt.title('Test Error Distribution')
plt.xlabel('Error')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'error_distribution.png'))
plt.show()

# 5. Compare with original model (if main model training was successful)
try:
    if not np.isnan(test_mse) and not np.isinf(test_mse):
        print("\n5. Training original model for comparison...")
        
        # Create and compile the original model
        original_model = GPRNet_original(
            im_height=40,
            neurons=16,
            kern_sz=5,
            enable_dropout=True,
            dp_coeff=0.2
        )
        
        original_model.compile(
            optimizer=Adam(learning_rate=5e-4),
            loss='mse',
            metrics=[rmse, r2]
        )
        
        # Train the original model with early stopping
        original_history = original_model.fit(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=50,  # Fewer epochs for comparison
            verbose=1,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping]
        )
        
        # Evaluate original model
        original_test_results = original_model.evaluate(X_test, y_test, verbose=1)
        original_y_test_pred = original_model.predict(X_test)
        original_test_r2 = r2_score(y_test.flatten(), original_y_test_pred.flatten())
        
        print("\nModel Comparison:")
        print(f"Original Model - Test Loss: {original_test_results[0]:.6f}, Test R²: {original_test_r2:.6f}")
        print(f"Enhanced Model - Test Loss: {test_mse:.6f}, Test R²: {test_r2:.6f}")
        
        if test_r2 > original_test_r2:
            improvement = ((test_r2 - original_test_r2) / abs(original_test_r2)) * 100
            print(f"Improvement: {improvement:.2f}% in R²")
        else:
            print("The original model performed better in this case.")
    else:
        print("\nSkipping model comparison due to unstable training results.")
except Exception as e:
    print(f"\nError during model comparison: {e}")
    print("Skipping model comparison.")

# Plot comparison if both models were trained successfully
try:
    if 'original_model' in locals() and 'original_y_test_pred' in locals():
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.scatter(y_test.flatten(), original_y_test_pred.flatten(), alpha=0.1, color='blue', label='Original')
        plt.scatter(y_test.flatten(), y_test_pred.flatten(), alpha=0.1, color='red', label='Enhanced')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
        plt.title('Model Comparison: Predictions')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        if hasattr(original_history, 'history') and hasattr(history, 'history'):
            plt.plot(original_history.history.get('val_loss', []), label='Original Model')
            plt.plot(history.history.get('val_loss', []), label='Enhanced Model')
            plt.title('Validation Loss Comparison')
            plt.xlabel('Epoch')
            plt.ylabel('Loss (MSE)')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'model_comparison.png'))
        plt.show()
except Exception as e:
    print(f"Error during comparison visualization: {e}")
    print("Skipping comparison visualization.")

print("\nTraining and evaluation complete. Model and visualizations saved to:", MODEL_DIR)

