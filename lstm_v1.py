import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# 1. Define File Paths
train_path = '/home/harsh/NIDS_ANN/KDDTrain+.txt'
test_path = '/home/harsh/NIDS_ANN/KDDTest+.txt'

# 2. Define Column Names 
columns = (['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 
            'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 
            'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
            'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
            'num_access_files', 'num_outbound_cmds', 'is_host_login', 
            'is_guest_login', 'count', 'srv_count', 'serror_rate', 
            'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
            'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 
            'dst_host_srv_count', 'dst_host_same_srv_rate', 
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
            'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'])

# 3. Load Datasets
train_df = pd.read_csv(train_path, names=columns)
test_df = pd.read_csv(test_path, names=columns)

# Drop 'difficulty'
train_df.drop('difficulty', axis=1, inplace=True)
test_df.drop('difficulty', axis=1, inplace=True)

# 4. Map Specific Attacks
dos_attacks = ['apache2', 'back', 'land', 'neptune', 'mailbomb', 'pod', 'processtable', 'smurf', 'teardrop', 'udpstorm', 'worm']
probe_attacks = ['ipsweep', 'mscan', 'nmap', 'portsweep', 'saint', 'satan']
u2r_attacks = ['buffer_overflow', 'loadmodule', 'perl', 'ps', 'rootkit', 'sqlattack', 'xterm']
r2l_attacks = ['ftp_write', 'guess_passwd', 'httptunnel', 'imap', 'multihop', 'named', 'phf', 'sendmail', 'snmpgetattack', 'snmpguess', 'spy', 'warezclient', 'warezmaster', 'xlock', 'xsnoop']

def map_attack(attack):
    if attack == 'normal':
        return 'Normal'
    elif attack in dos_attacks:
        return 'DoS'
    elif attack in probe_attacks:
        return 'Probe'
    elif attack in u2r_attacks:
        return 'U2R'
    elif attack in r2l_attacks:
        return 'R2L'
    else:
        return 'Unknown'

train_df['attack_type'] = train_df['attack_type'].apply(map_attack)
test_df['attack_type'] = test_df['attack_type'].apply(map_attack)

# 5. Split Features and Target
X_train = train_df.drop('attack_type', axis=1)
y_train = train_df['attack_type']
X_test = test_df.drop('attack_type', axis=1)
y_test = test_df['attack_type']

# 6. Data Preprocessing
categorical_cols = ['protocol_type', 'service', 'flag']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', MinMaxScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
    ])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# --- APPLY SMOTE ---
print("Applying SMOTE to balance the training data...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_encoded)

y_train_resampled_categorical = tf.keras.utils.to_categorical(y_train_resampled, num_classes=5)
y_test_categorical = tf.keras.utils.to_categorical(y_test_encoded, num_classes=5)

# --- NEW SECTION: RESHAPE FOR LSTM ---
# LSTMs expect 3D input: (Samples, Timesteps, Features)
print("Reshaping data for LSTM 3D input requirements...")
input_dim = X_train_resampled.shape[1]

X_train_lstm = np.reshape(X_train_resampled, (X_train_resampled.shape[0], 1, input_dim))
X_test_lstm = np.reshape(X_test_processed, (X_test_processed.shape[0], 1, input_dim))
# ------------------------------------

# 7. Build the LSTM Architecture
model = Sequential()
# First LSTM layer. 'return_sequences=True' is required when stacking LSTMs.
model.add(LSTM(64, input_shape=(1, input_dim), activation='tanh', return_sequences=True))
model.add(Dropout(0.5))
# Second LSTM layer acting as the bottleneck.
model.add(LSTM(32, activation='tanh'))
model.add(Dense(5, activation='softmax'))

# 8. Compile the Model
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy'])

# 9. Train the Model
print("\nStarting LSTM model training...")
history = model.fit(X_train_lstm, y_train_resampled_categorical, 
                    epochs=20, 
                    batch_size=256, 
                    validation_split=0.1, 
                    verbose=1)

# 10. Evaluate the Model
print("\nEvaluating LSTM model on test data...")
loss, accuracy = model.evaluate(X_test_lstm, y_test_categorical, verbose=0)
print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")

# 11. Generate Classification Report
y_pred = model.predict(X_test_lstm)
y_pred_classes = np.argmax(y_pred, axis=1)

print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_classes, target_names=label_encoder.classes_))