# This was part of an experiment to test if 
# combining different types of supervised models could be used collectively 
# to improve accuracy of the results.
# 
# While it did improve the results by a slight degree, it is too computationally expensive for 
# weak hardware like a raspberry pi.  


# import json
# import os

# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from joblib import dump, load
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import GridSearchCV, train_test_split
# from sklearn.preprocessing import StandardScaler
# from tensorflow.keras.models import load_model

# script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = os.path.dirname(script_dir)


# NON_PLASTIC_DATA_FILE = f"{script_dir}/training_data/with_glass/non_plastic_data.json"
# PLASTIC_DATA_FILE = f"{script_dir}/training_data/with_glass/plastic_data.json"
# TRAINED_MODEL_FILE = f"{script_dir}/trained_model.joblib"
# TRAINED_NN_MODEL_FILE = f"{script_dir}/trained_nn_model"


# def load_json_lines(filename):
#     """Load JSON lines from file and skip improper lines."""
#     data = []
#     with open(filename, "r") as file:
#         for line in file:
#             try:
#                 line = line.strip().rstrip(",")
#                 if line and not line.startswith("//"):
#                     json_obj = json.loads(line)
#                     data.append(json_obj)
#             except json.JSONDecodeError:
#                 print(f"Skipping line due to JSONDecodeError: {line}")
#     return data


# def load_model(TRAINED_MODEL_FILE):
#     """Check for a saved model and load it."""
#     try:
#         model = load(TRAINED_MODEL_FILE)
#         print("Model loaded successfully.")
#         return model, True
#     except FileNotFoundError:
#         print("No pre-trained model found. Proceeding to train a new model.")
#         return None, False


# def build_compile_model(input_shape):
#     """Build and compile the neural network model."""
#     model = tf.keras.Sequential(
#         [
#             tf.keras.layers.Dense(128, activation="relu", input_shape=(input_shape,)),
#             tf.keras.layers.Dropout(0.2),
#             tf.keras.layers.Dense(64, activation="relu"),
#             tf.keras.layers.Dense(1, activation="sigmoid"),
#         ]
#     )

#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model


# def train_and_predict_with_nn(sample, data):
#     trained_model_dir = TRAINED_NN_MODEL_FILE  # This should be a directory path

#     # Prepare the dataset
#     X = data.drop("is_plastic", axis=1)
#     y = data["is_plastic"]
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)

#     # Split the dataset
#     X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(
#         X_scaled, y, test_size=0.2, random_state=42
#     )

#     # Load or train the model
#     if os.path.exists(trained_model_dir) and len(os.listdir(trained_model_dir)) > 0:
#         print("Loading trained neural network model...")
#         model = tf.keras.models.load_model(trained_model_dir)
#     else:
#         print("Training neural network model...")
#         model = build_compile_model(X_train_scaled.shape[1])
#         model.fit(
#             X_train_scaled,
#             y_train,
#             epochs=10,
#             validation_split=0.2,
#             batch_size=10,
#             verbose=1,
#         )
#         model.save(trained_model_dir)

#     # Preprocess the sample
#     sample_scaled = scaler.transform(sample)

#     # Predict the sample's probability
#     probability = model.predict(sample_scaled)
#     return probability[0][0]


# def combined_predict(sample):
#     best_model, model_loaded = load_model(TRAINED_MODEL_FILE)

#     non_plastics_list = load_json_lines(NON_PLASTIC_DATA_FILE)
#     plastics_list = load_json_lines(PLASTIC_DATA_FILE)

#     non_plastics_df = pd.DataFrame(non_plastics_list).drop_duplicates()
#     plastics_df = pd.DataFrame(plastics_list).drop_duplicates()

#     if len(plastics_df) < len(non_plastics_df):
#         length_difference = len(non_plastics_df) - len(plastics_df)
#         print(
#             f"Balancing non_plastics to plastics: Ignoring {length_difference} samples."
#         )
#         non_plastics_df = non_plastics_df.iloc[: len(plastics_df)]

#     non_plastics = non_plastics_df
#     plastics = plastics_df

#     non_plastics["is_plastic"] = 0
#     plastics["is_plastic"] = 1

#     data = pd.concat([non_plastics, plastics], ignore_index=True)

#     imputer = SimpleImputer(strategy="mean")
#     data_imputed = pd.DataFrame(
#         imputer.fit_transform(data.drop(columns="is_plastic")),
#         columns=data.drop(columns="is_plastic").columns,
#     )
#     data_imputed["is_plastic"] = data["is_plastic"].values

#     X = data_imputed.drop("is_plastic", axis=1)
#     y = data_imputed["is_plastic"]

#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42
#     )

#     if not model_loaded:
#         param_grid = {
#             "n_estimators": [100, 200, 300],
#             "max_depth": [None, 10, 20, 30],
#             "min_samples_split": [2, 5, 10],
#             "min_samples_leaf": [1, 2, 4],
#         }
#         clf = RandomForestClassifier(random_state=42)
#         grid_search = GridSearchCV(
#             estimator=clf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2
#         )
#         grid_search.fit(X_train, y_train)

#         best_model = grid_search.best_estimator_

#         dump(best_model, TRAINED_MODEL_FILE)
#         print(f"Model trained and saved as {TRAINED_MODEL_FILE}")

#     y_pred = best_model.predict(X_test)

#     samples = []

#     if sample != "" and sample is not None:
#         sample1 = json.loads(sample)
#         samples.append(sample1)

#     if samples:
#         single_data_df = pd.DataFrame(samples)
#         # print(single_data_df)

#     sample_dict = json.loads(sample)

#     sample_df = pd.DataFrame([sample_dict])

#     sample_df = pd.DataFrame([sample_dict])
#     nn_probability = train_and_predict_with_nn(sample_df, data_imputed)
#     predicted_class_rf = best_model.predict(sample_df)
#     predicted_proba_rf = best_model.predict_proba(sample_df)

#     predicted_proba_rf = predicted_proba_rf[0][1]

#     def compute_percent_similarity(sample1, sample2):
#         similarities = []
#         for key in sample1:
#             if key in sample2:
#                 similarity = 1 / (1 + abs(sample1[key] - sample2[key]))
#                 similarities.append(similarity)
#             else:
#                 similarities.append(0)
#         return np.mean(similarities)

#     total_percent_similarities = []

#     for training_sample in plastics_list:
#         total_percent_similarity = compute_percent_similarity(
#             sample_dict, training_sample
#         )
#         total_percent_similarities.append(total_percent_similarity)

#     sample_data_similarity = np.mean(total_percent_similarities)
#     sample_data_similarity = np.clip(sample_data_similarity * 10, 0, 100)
#     combined_predictions = []

#     combined_predictions.append(predicted_proba_rf)
#     combined_predictions.append(sample_data_similarity)
#     combined_predictions.append(nn_probability)

#     final_prediction = np.mean(combined_predictions)
#     return final_prediction
