import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks, regularizers
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import joblib
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from sklearn.calibration import CalibratedClassifierCV
import google.generativeai as genai
from io import BytesIO
import base64
from streamlit_option_menu import option_menu

# Initialize Gemini API
genai.configure(api_key="AIzaSyD_TCm3Ubp4Wwq9XFQMALlnTFf2DmCfqSQ")
model = genai.GenerativeModel('gemini-1.5-flash')  # Updated to latest available model

# ========================================================
# 1. ENHANCED DATA LOADER WITH DATA AUGMENTATION
# ========================================================
class PneumoniaDataLoader:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path
        self.img_size = (224, 224)  # Increased for better feature extraction
        self.augmentor = self._build_augmentor()
        
    def _build_augmentor(self):
        """Build data augmentation pipeline with updated Keras API"""
        return tf.keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ])
        
    def load_dataset(self, dataset_type="train", augment=False):
        path = os.path.join(self.base_path, dataset_type)
        images = []
        labels = []
        file_paths = []
        
        class_mapping = {
            "NORMAL-TRAIN": 0,
            "NORMAL-TEST": 0,
            "PNEUMONIA-TRAIN": 1,
            "PNEUMONIA-TEST": 1
        }
        
        for class_dir in os.listdir(path):
            if not os.path.isdir(os.path.join(path, class_dir)):
                continue
                
            label = class_mapping.get(class_dir.upper(), -1)
            if label == -1:
                continue
                
            class_path = os.path.join(path, class_dir)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        raise ValueError(f"Could not read image {img_path}")
                        
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.img_size)
                    
                    # Apply augmentation only to training data
                    if augment and dataset_type == "train":
                        img = self.augmentor(tf.expand_dims(img, axis=0)).numpy()[0]
                    
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
                    file_paths.append(img_path)
                except Exception as e:
                    print(f"Skipping {img_path}: {str(e)}")
        
        return np.array(images), np.array(labels), file_paths

# ========================================================
# 2. ADVANCED CLASSIFICATION SYSTEM
# ========================================================
class PneumoniaClassifier:
    def __init__(self):
        self.img_size = (224, 224)
        self.cnn_model = self._build_advanced_cnn()
        self.sklearn_model = None
        self.train_history = {
            'train_acc': [],
            'val_acc': [],
            'train_loss': [],
            'val_loss': []
        }
        
    def _build_advanced_cnn(self):
        """Build a more sophisticated CNN with transfer learning"""
        base_model = EfficientNetB0(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:100]:
            layer.trainable = False
            
        # Add custom head
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        predictions = Dense(1, activation='sigmoid')(x)
        
        model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_cnn(self, X_train, y_train, X_val, y_val, epochs=20, batch_size=32):
        """Train the CNN with early stopping"""
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        history = self.cnn_model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Store history
        self.train_history['train_acc'] = history.history['accuracy']
        self.train_history['val_acc'] = history.history['val_accuracy']
        self.train_history['train_loss'] = history.history['loss']
        self.train_history['val_loss'] = history.history['val_loss']
        
        return history
    
    def train_ml_model(self, X_train, y_train, X_val, y_val, model_type='gbm'):
        """Train a machine learning model on CNN features"""
        # Extract features
        train_features = self.cnn_model.predict(X_train, verbose=0)
        val_features = self.cnn_model.predict(X_val, verbose=0)
        
        # Select model
        if model_type == 'gbm':
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                random_state=42
            )
        elif model_type == 'svm':
            model = CalibratedClassifierCV(
                SVC(kernel='rbf', C=1.0, probability=True),
                method='sigmoid'
            )
        else:  # random forest
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
        
        # Train
        model.fit(train_features, y_train)
        
        # Evaluate on validation
        val_acc = model.score(val_features, y_val)
        print(f"{model_type.upper()} Validation Accuracy: {val_acc:.4f}")
        
        self.sklearn_model = model
        return val_acc
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        # Get CNN features
        test_features = self.cnn_model.predict(X_test, verbose=0)
        
        # Get predictions
        if hasattr(self.sklearn_model, 'predict_proba'):
            y_proba = self.sklearn_model.predict_proba(test_features)
            y_pred = self.sklearn_model.predict(test_features)
        else:
            y_pred = self.sklearn_model.predict(test_features)
            y_proba = np.zeros((len(y_pred), 2))
            y_proba[:, 1] = y_pred
            y_proba[:, 0] = 1 - y_pred
        
        return y_test, y_pred, y_proba
    
    def predict_single(self, image):
        """Predict on a single image"""
        img = np.array(image.convert('RGB'))
        img = cv2.resize(img, self.img_size)
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Get CNN features
        features = self.cnn_model.predict(img, verbose=0)
        
        # Get prediction
        if hasattr(self.sklearn_model, 'predict_proba'):
            proba = self.sklearn_model.predict_proba(features)[0]
            pred = self.sklearn_model.predict(features)[0]
        else:
            pred = self.sklearn_model.predict(features)[0]
            proba = np.array([1-pred, pred])  # Dummy probabilities
        
        return pred, proba

# ========================================================
# 3. GEMINI AI INTERACTION FUNCTIONS
# ========================================================
def get_gemini_response(image, diagnosis, confidence):
    """Get response from Gemini AI based on the diagnosis"""
    try:
        # Convert image to base64 for Gemini
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        prompt = f"""
        You are a medical assistant specialized in respiratory diseases. 
        A patient has uploaded a chest X-ray with the following diagnosis:
        
        Diagnosis: {'Pneumonia detected' if diagnosis == 1 else 'No pneumonia detected'}
        Confidence: {confidence:.2f}%
        
        Please provide:
        1. A clear explanation of what this means in simple terms
        2. Recommended next steps
        3. When to seek immediate medical attention
        4. General prevention tips for respiratory health
        
        Be compassionate and professional, but avoid alarming language.
        """
        
        response = model.generate_content(
            contents=[prompt, {"mime_type": "image/jpeg", "data": img_str}]
        )
        
        return response.text
    except Exception as e:
        return f"Could not get AI response: {str(e)}"

def chat_with_gemini():
    """Interactive chat interface with Gemini about pneumonia"""
    st.subheader("Pneumonia Information Chat")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about pneumonia..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get Gemini response
        try:
            response = model.generate_content(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response.text})
        except Exception as e:
            st.error(f"Error getting response: {str(e)}")

# ========================================================
# 4. STREAMLIT APP WITH ADVANCED VISUALIZATIONS
# ========================================================
def main():
    st.set_page_config(
        layout="wide",
        page_title="Advanced Pneumonia Classification",
        page_icon="🩺"
    )
    
    # Custom CSS for improved UI
    st.markdown("""
    <style>
    .stApp {
        background-color: #f5f5f5;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    .st-b7 {
        background-color: #ffffff;
    }
    .st-c0 {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 24px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stFileUploader>div>div>div>div {
        color: #4CAF50;
        border-color: #4CAF50;
    }
    .stSelectbox>div>div>div {
        color: #4CAF50;
    }
    .stSlider>div>div>div {
        color: #4CAF50;
    }
    .css-1aumxhk {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Navigation bar with icons
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Diagnose", "Chatbot", "About"],
            icons=["house", "clipboard-pulse", "robot", "info-circle"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#ffffff"},
                "icon": {"color": "#4CAF50", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#e0f7fa"},
                "nav-link-selected": {"background-color": "#4CAF50"},
            }
        )
    
    # Initialize session state
    if 'classifier' not in st.session_state:
        st.session_state.classifier = PneumoniaClassifier()
        st.session_state.trained = False
        st.session_state.test_images = None
        st.session_state.test_labels = None
        st.session_state.test_paths = None
        st.session_state.train_history = None
    
    if selected == "Home":
        st.title("Advanced Pneumonia Classification from Chest X-rays")
        
        # Main content area
        tab1, tab2 = st.tabs(["Diagnosis Dashboard", "Model Information"])
        
        with tab1:
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.header("Image Analysis")
                if st.session_state.test_paths:
                    selected_idx = st.selectbox(
                        "Select test image to view",
                        range(len(st.session_state.test_paths)),
                        format_func=lambda x: os.path.basename(st.session_state.test_paths[x])
                    )
                    
                    if selected_idx is not None:
                        try:
                            img = Image.open(st.session_state.test_paths[selected_idx])
                            st.image(img, caption=f"Selected: {os.path.basename(st.session_state.test_paths[selected_idx])}")
                            
                            if st.session_state.trained:
                                pred, proba = st.session_state.classifier.predict_single(img)
                                actual_label = "Normal" if st.session_state.test_labels[selected_idx] == 0 else "Pneumonia"
                                pred_label = "Normal" if pred == 0 else "Pneumonia"
                                
                                st.write(f"**Actual:** {actual_label}")
                                st.write(f"**Predicted:** {pred_label}")
                                
                                # Visualize prediction confidence
                                fig, ax = plt.subplots(figsize=(8, 4))
                                ax.barh(["Normal", "Pneumonia"], proba, color=['green', 'red'])
                                ax.set_xlim(0, 1)
                                ax.set_xlabel("Probability")
                                ax.set_title("Prediction Confidence")
                                for i, v in enumerate(proba):
                                    ax.text(v + 0.01, i, f"{v:.2f}", color='black', fontweight='bold')
                                st.pyplot(fig)
                        except Exception as e:
                            st.error(f"Error displaying image: {str(e)}")
            
            with col2:
                if st.session_state.trained:
                    st.header("Model Performance Analysis")
                    
                    # CNN Training History
                    st.subheader("CNN Training Progress")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
                    
                    # Accuracy plot
                    ax1.plot(st.session_state.train_history['train_acc'], label='Training Accuracy')
                    ax1.plot(st.session_state.train_history['val_acc'], label='Validation Accuracy')
                    ax1.set_title('Model Accuracy')
                    ax1.set_ylabel('Accuracy')
                    ax1.set_xlabel('Epoch')
                    ax1.legend()
                    
                    # Loss plot
                    ax2.plot(st.session_state.train_history['train_loss'], label='Training Loss')
                    ax2.plot(st.session_state.train_history['val_loss'], label='Validation Loss')
                    ax2.set_title('Model Loss')
                    ax2.set_ylabel('Loss')
                    ax2.set_xlabel('Epoch')
                    ax2.legend()
                    
                    st.pyplot(fig)
                    
                    # Classification Metrics
                    st.subheader("Classification Report")
                    report = classification_report(
                        st.session_state.y_true,
                        st.session_state.y_pred,
                        target_names=["Normal", "Pneumonia"],
                        output_dict=True
                    )
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))
                    
                    # Confusion Matrix and ROC Curve
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Confusion Matrix**")
                        cm = confusion_matrix(st.session_state.y_true, st.session_state.y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                                    xticklabels=["Normal", "Pneumonia"],
                                    yticklabels=["Normal", "Pneumonia"])
                        ax.set_xlabel("Predicted")
                        ax.set_ylabel("Actual")
                        st.pyplot(fig)
                        
                        # Key Metrics
                        st.markdown("**Performance Metrics**")
                        accuracy = np.mean(st.session_state.y_true == st.session_state.y_pred)
                        precision = report['Pneumonia']['precision']
                        recall = report['Pneumonia']['recall']
                        f1 = report['Pneumonia']['f1-score']
                        
                        st.metric("Accuracy", f"{accuracy:.2%}")
                        st.metric("Precision", f"{precision:.2%}")
                        st.metric("Recall", f"{recall:.2%}")
                        st.metric("F1 Score", f"{f1:.2%}")
                    
                    with col2:
                        st.markdown("**ROC Curve**")
                        fpr, tpr, _ = roc_curve(st.session_state.y_true, st.session_state.y_proba[:, 1])
                        roc_auc = auc(fpr, tpr)
                        
                        fig, ax = plt.subplots()
                        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                                label=f'ROC curve (AUC = {roc_auc:.2f})')
                        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                        ax.set_xlim([0.0, 1.0])
                        ax.set_ylim([0.0, 1.05])
                        ax.set_xlabel('False Positive Rate')
                        ax.set_ylabel('True Positive Rate')
                        ax.set_title('Receiver Operating Characteristic')
                        ax.legend(loc="lower right")
                        st.pyplot(fig)
                        
                        # Probability Distribution
                        st.markdown("**Probability Distribution**")
                        proba_df = pd.DataFrame({
                            "True Label": ["Normal" if x == 0 else "Pneumonia" 
                                        for x in st.session_state.y_true],
                            "Pneumonia Probability": st.session_state.y_proba[:, 1]
                        })
                        fig = plt.figure(figsize=(10, 4))
                        sns.histplot(
                            data=proba_df, 
                            x="Pneumonia Probability", 
                            hue="True Label",
                            element="step",
                            stat="density",
                            common_norm=False,
                            bins=20
                        )
                        plt.title("Prediction Probability Distribution")
                        st.pyplot(fig)
                else:
                    st.warning("Please configure and train the model using the sidebar controls")
        
        with tab2:
            st.header("Model Information")
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <h3 style='color: #4CAF50;'>Model Architecture</h3>
                <p>This application uses a hybrid deep learning approach combining:</p>
                <ul>
                    <li><strong>EfficientNetB0</strong> as a feature extractor (transfer learning)</li>
                    <li>Custom dense layers for classification</li>
                    <li>Machine learning classifiers (GBM, Random Forest, or SVM) on top of CNN features</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: #4CAF50;'>Training Parameters</h3>
                <ul>
                    <li><strong>Input Size:</strong> 224x224 pixels (RGB)</li>
                    <li><strong>Augmentation:</strong> Random rotation, zoom, and contrast</li>
                    <li><strong>Regularization:</strong> L2 regularization and dropout</li>
                    <li><strong>Optimizer:</strong> Adam with learning rate 1e-4</li>
                    <li><strong>Early Stopping:</strong> Patience of 5 epochs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    elif selected == "Diagnose":
        st.title("Diagnose Your X-ray")
        
        # Create a card-like container for the uploader
        with st.container():
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: #4CAF50;'>Upload Your Chest X-ray</h3>
                <p>Get an instant pneumonia diagnosis by uploading your chest X-ray image</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], key="diagnose_uploader")
            
            if uploaded_file is not None and st.session_state.trained:
                try:
                    image = Image.open(uploaded_file)
                    
                    # Display in a card
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(image, caption="Uploaded X-ray", use_column_width=True)
                    
                    with col2:
                        with st.spinner("Analyzing image..."):
                            pred, proba = st.session_state.classifier.predict_single(image)
                            confidence = proba[1] if pred == 1 else proba[0]
                            
                            if pred == 1:
                                st.error(f"Pneumonia detected (Confidence: {confidence*100:.2f}%)")
                            else:
                                st.success(f"No pneumonia detected (Confidence: {confidence*100:.2f}%)")
                            
                            # Get Gemini AI explanation
                            with st.spinner("Getting medical insights..."):
                                ai_response = get_gemini_response(image, pred, confidence*100)
                                st.markdown("### Medical Insights")
                                st.markdown(ai_response)
                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")
    
    elif selected == "Chatbot":
        st.title("Pneumonia Information Chatbot")
        chat_with_gemini()
    
    elif selected == "About":
        st.title("About This Application")
        
        # Create a nice about section with cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <h3 style='color: #4CAF50;'>Application Purpose</h3>
                <p>This application uses advanced deep learning to detect pneumonia from chest X-ray images. 
                It combines convolutional neural networks with machine learning classifiers for accurate diagnosis.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <h3 style='color: #4CAF50;'>Technology Stack</h3>
                <ul>
                    <li>TensorFlow/Keras for deep learning</li>
                    <li>EfficientNetB0 as feature extractor</li>
                    <li>Scikit-learn for classification</li>
                    <li>Streamlit for user interface</li>
                    <li>Google Gemini for medical insights</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 20px;'>
                <h3 style='color: #4CAF50;'>Disclaimer</h3>
                <p>This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment.</p>
                <p>Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style='background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);'>
                <h3 style='color: #4CAF50;'>Contact Information</h3>
                <p>For questions or feedback about this application, please contact:</p>
                <p>Email: support@medicalai.com</p>
                <p>Phone: (123) 456-7890</p>
            </div>
            """, unsafe_allow_html=True)

    # Sidebar controls (same as before)
    with st.sidebar:
        if selected in ["Home", "Diagnose"]:
            st.header("Model Configuration")
            
            model_type = st.selectbox(
                "Select ML Model",
                ["Gradient Boosting", "Random Forest", "SVM"],
                index=0
            )
            
            epochs = st.slider("CNN Epochs", 5, 50, 20, 1)
            batch_size = st.slider("Batch Size", 16, 128, 32, 16)
            validation_split = st.slider("Validation Split", 0.1, 0.3, 0.2, 0.05)
            augment_data = st.checkbox("Use Data Augmentation", value=True)
            
            if st.button("Train and Evaluate Model"):
                with st.spinner("Loading and preprocessing data..."):
                    loader = PneumoniaDataLoader()
                    try:
                        X_train, y_train, _ = loader.load_dataset("train", augment=augment_data)
                        X_test, y_test, test_paths = loader.load_dataset("test")
                        
                        if len(X_train) == 0 or len(X_test) == 0:
                            st.error("No images found in the dataset directory. Please check your dataset path.")
                            return
                        
                        # Split into train/validation
                        X_train, X_val, y_train, y_val = train_test_split(
                            X_train, y_train,
                            test_size=validation_split,
                            random_state=42,
                            stratify=y_train
                        )
                        
                        st.session_state.test_images = X_test
                        st.session_state.test_labels = y_test
                        st.session_state.test_paths = test_paths
                        
                        with st.spinner("Training CNN..."):
                            # Train CNN
                            cnn_history = st.session_state.classifier.train_cnn(
                                X_train, y_train,
                                X_val, y_val,
                                epochs=epochs,
                                batch_size=batch_size
                            )
                            
                            # Train ML model
                            ml_model_type = model_type.lower().replace(" ", "_")
                            with st.spinner(f"Training {model_type}..."):
                                val_acc = st.session_state.classifier.train_ml_model(
                                    X_train, y_train,
                                    X_val, y_val,
                                    model_type=ml_model_type
                                )
                            
                            st.session_state.trained = True
                            st.session_state.train_history = st.session_state.classifier.train_history
                            
                            # Evaluate
                            y_true, y_pred, y_proba = st.session_state.classifier.evaluate(X_test, y_test)
                            
                            # Save evaluation results
                            st.session_state.y_true = y_true
                            st.session_state.y_pred = y_pred
                            st.session_state.y_proba = y_proba
                            
                            st.success("Training and evaluation complete!")
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()