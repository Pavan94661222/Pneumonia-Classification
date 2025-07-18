# **Pneumonia Detection from Chest X-rays**  

This project leverages **EfficientNetB0**, a powerful pre-trained CNN, for feature extraction from chest X-ray images using transfer learning. The extracted features are then fed into a **Gradient Boosting Classifier**, an ensemble learning method, to improve diagnostic accuracy. The hybrid approach combines deep learning for image understanding with robust machine learning for classification.  

The system provides an interactive **Streamlit-based dashboard** for uploading X-rays, visualizing predictions, and generating AI-powered medical insights. It includes performance metrics such as accuracy, precision, recall, and ROC curves to evaluate model effectiveness.  

Key features include **data augmentation** (rotation, zoom, contrast adjustments) to enhance model generalization and **probability calibration** for reliable confidence scores. The application also integrates an **AI chatbot** for explaining diagnoses and suggesting next steps.  

Built for **educational and research purposes**, this tool demonstrates how deep learning and ensemble methods can assist in medical imaging analysis. It is not a substitute for professional diagnosis but serves as a proof-of-concept for AI in healthcare.  

The project is implemented in **Python** using **TensorFlow/Keras** for deep learning and **Scikit-learn** for machine learning, with an intuitive web interface for easy interaction.
