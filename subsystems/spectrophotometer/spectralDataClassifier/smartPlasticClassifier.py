# This was part of an experiment to test if 
# combining different types of supervised models could be used collectively 
# to improve accuracy of the results.
# 
# While it did improve the results by a slight degree, it is too computationally expensive for 
# weak hardware like a raspberry pi.  


# import json
# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, classification_report
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from joblib import dump, load
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


# # Function to load JSON lines from file and skip improper lines
# def load_json_lines(filename):
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


# # Function to check for a saved model and load it
# def load_model(model_filename):
#     try:
#         model = load(model_filename)
#         print("Model loaded successfully.")
#         return model, True
#     except FileNotFoundError:
#         print("No pre-trained model found. Proceeding to train a new model.")
#         return None, False


# # Define file paths
# script_dir = os.path.dirname(os.path.abspath(__file__))
# model_filename = f"{script_dir}/trained_model.joblib"

# # Load or check for existing trained model
# best_model, model_loaded = load_model(model_filename)

# # Load and process data
# non_plastics_list = load_json_lines(
#     os.path.join(script_dir, "training_data/with_glass/non_plastic_data.json")
# )
# plastics_list = load_json_lines(
#     os.path.join(script_dir, "training_data/with_glass/plastic_data.json")
# )

# non_plastics_df = pd.DataFrame(non_plastics_list).drop_duplicates()
# plastics_df = pd.DataFrame(plastics_list).drop_duplicates()

# min_samples = min(len(non_plastics_df), len(plastics_df))
# non_plastics_df = non_plastics_df.sample(n=min_samples)
# plastics_df = plastics_df.sample(n=min_samples)

# non_plastics_df["is_plastic"] = 0
# plastics_df["is_plastic"] = 1

# data = pd.concat([non_plastics_df, plastics_df], ignore_index=True)

# # Handle missing values
# imputer = SimpleImputer(strategy="mean")
# data_imputed = pd.DataFrame(
#     imputer.fit_transform(data.drop(columns="is_plastic")),
#     columns=data.drop(columns="is_plastic").columns,
# )
# data_imputed["is_plastic"] = data["is_plastic"].values

# X = data_imputed.drop("is_plastic", axis=1)
# y = data_imputed["is_plastic"]

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# # Define pipeline with preprocessing steps and classifiers
# pipeline = Pipeline(
#     [
#         ("imputer", SimpleImputer(strategy="mean")),
#         ("scaler", StandardScaler()),
#         ("ensemble", RandomForestClassifier(random_state=42)),
#     ]
# )

# # Hyperparameter tuning with cross-validation for RandomForestClassifier
# param_grid_rf = {
#     "ensemble__n_estimators": [100, 200, 300],
#     "ensemble__max_depth": [None, 10, 20, 30],
#     "ensemble__min_samples_split": [2, 5, 10],
#     "ensemble__min_samples_leaf": [1, 2, 4],
# }

# grid_search_rf = GridSearchCV(
#     estimator=pipeline, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2
# )
# grid_search_rf.fit(X_train, y_train)

# best_model_rf = grid_search_rf.best_estimator_


# # Define function to create neural network model
# def create_nn_model():
#     model = Sequential(
#         [
#             Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
#             Dropout(0.5),
#             Dense(64, activation="relu"),
#             Dropout(0.5),
#             Dense(1, activation="sigmoid"),
#         ]
#     )
#     model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#     return model


# # Create KerasClassifier wrapper for scikit-learn
# nn_model = KerasClassifier(build_fn=create_nn_model, verbose=0)

# # Train neural network model
# history = nn_model.fit(
#     X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1
# )

# # Save or load trained models
# if not model_loaded:
#     dump(best_model_rf, model_filename)
#     print(f"Random Forest model trained and saved as {model_filename}")
#     nn_model.model.save(f"{script_dir}/trained_model_nn.h5")
#     print("Neural network model trained and saved as trained_model_nn.h5")
# else:
#     print("Best models already loaded.")

# # Evaluate models performance
# y_pred_rf = best_model_rf.predict(X_test)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)
# print(f"Random Forest Model Accuracy: {accuracy_rf}")
# print("Random Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf))

# _, accuracy_nn = nn_model.model.evaluate(X_test, y_test, verbose=0)
# print(f"Neural Network Model Accuracy: {accuracy_nn}")

# # Handling sample predictions
# confirmed_plastic = '{"A":0.87,"B":12.54,"C":118.20,"D":41.07,"E":27.13,"F":14.71,"G":3.53,"H":3.34,"R":7.03,"I":4.47,"S":1.96,"J":0.89,"T":0.79,"U":0.81,"V":0.86,"W":1.06,"K":0.73,"L":0.00}'
# sample_dict = json.loads(confirmed_plastic)
# sample_df = pd.DataFrame([sample_dict])

# # Predict the class of the sample using Random Forest classifier
# predicted_class_rf = best_model_rf.predict(sample_df)
# predicted_proba_rf = best_model_rf.predict_proba(sample_df)

# # Convert probabilities to percentages
# predicted_proba_percent_rf = [f"{prob * 100:.4f}" for prob in predicted_proba_rf[0]]

# # Assign labels based on predicted class
# predicted_class_label_rf = "PET Plastic" if predicted_class_rf[0] == 0 else "Not PET"

# print("Random Forest Classifier Prediction:")
# print(f"Predicted Class: {predicted_class_label_rf}")
# print(f"RFC Probability Prediction: {predicted_proba_percent_rf[0]}%")

# # Predict the class of the sample using Neural Network
# predicted_proba_nn = nn_model.predict_proba(sample_df)
# predicted_class_nn = 1 if predicted_proba_nn >= 0.5 else 0

# print("Neural Network Classifier Prediction:")
# print(f"Predicted Class: {'Not PET' if predicted_class_nn == 1 else 'PET Plastic'}")
# print(f"NN Probability Prediction: {predicted_proba_nn[0][0] * 100:.4f}%")

# final_prediction = (predicted_proba_nn + predicted_class_rf) / 2

# print(f"Final Prediction: {final_prediction*100:.4f}%")
