# 📩 SMS/Email Spam Classifier

A machine learning-based spam detection system that classifies SMS and email messages as either **Spam** or **Ham** (Not Spam) using Naive Bayes algorithms and Natural Language Processing techniques.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange)

## 📌 Project Overview

This project implements a comprehensive Spam Detection System using three variants of the Naive Bayes algorithm:
- **GaussianNB** - Assumes features follow a normal distribution
- **MultinomialNB** - Best for discrete counts such as word frequencies
- **BernoulliNB** - Suitable for binary/boolean features

The system uses probabilistic learning based on Bayes' theorem to accurately classify messages with high precision and recall.

## ✨ Features

- 🎯 **High Accuracy Detection** - Up to 98% accuracy with MultinomialNB
- 🔄 **Text Preprocessing** - Advanced cleaning, tokenization, and vectorization
- 📊 **Model Comparison** - Side-by-side evaluation of three Naive Bayes variants
- 🌐 **Web Interface** - User-friendly Flask application for real-time predictions
- 💾 **Pre-trained Models** - Ready-to-use serialized models for instant deployment
- 📈 **Performance Metrics** - Comprehensive evaluation using accuracy, precision, recall, and F1-score

## 🛠️ Tech Stack

- **Programming Language:** Python 3.8+
- **Web Framework:** Flask
- **ML Libraries:** scikit-learn, pandas, numpy
- **NLP Tools:** NLTK, TF-IDF Vectorization
- **Data Visualization:** Matplotlib, Seaborn
- **Model Serialization:** Pickle

## 📂 Project Structure

```
SMS-Email---Spam---Classifier/
│
├── Sms_Spam_Detection.ipynb    # Jupyter notebook with model training
├── app.py                       # Flask web application
├── model.pkl                    # Pre-trained ML classifier model
├── vectorizer.pkl               # Pre-trained TF-IDF vectorizer
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies
```

## 📊 Dataset

The project uses the **SMS Spam Collection Dataset** containing SMS messages labeled as "spam" or "ham".

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- **Total Messages:** ~5,572
- **Distribution:**
  - Ham (legitimate): ~87%
  - Spam: ~13%

## 🧠 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **GaussianNB** | 0.86 | 0.65 | 0.78 | 0.71 |
| **MultinomialNB** ⭐ | **0.98** | **0.96** | **0.94** | **0.95** |
| **BernoulliNB** | 0.97 | 0.95 | 0.92 | 0.93 |

✅ **Best Model:** MultinomialNB offers the best balance of precision and recall for spam detection.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/subhadip2510/SMS-Email---Spam---Classifier.git
   cd SMS-Email---Spam---Classifier
   ```

2. **Create a virtual environment (recommended)**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data (if required)**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   ```

### Running the Application

1. **Start the Flask server**
   ```bash
   python app.py
   ```

2. **Access the application**
   - Open your web browser
   - Navigate to `http://127.0.0.1:5000/`

3. **Test the classifier**
   - Enter an SMS or email message in the text area
   - Click the "Predict" button
   - View the classification result (Spam or Ham)

## 💡 Usage Examples

### Example Messages

**Spam Example:**
```
Congratulations! You've won a $1000 gift card. Click here to claim now!
```
**Result:** ⚠️ SPAM

**Ham Example:**
```
Hey, are we still meeting for lunch tomorrow at 1pm?
```
**Result:** ✅ HAM (Not Spam)

### Using the Model Programmatically

```python
import pickle

# Load the pre-trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Predict a message
message = "Free entry! Win a prize worth $5000. Text WIN to 12345"
transformed_message = vectorizer.transform([message])
prediction = model.predict(transformed_message)

print("Spam" if prediction[0] == 1 else "Ham")
```

## 🔍 How It Works

### 1. Data Preprocessing
- Remove special characters and punctuation
- Convert text to lowercase
- Tokenization
- Remove stop words
- Apply stemming/lemmatization

### 2. Feature Extraction
- **TF-IDF Vectorization** (Term Frequency-Inverse Document Frequency)
- Converts text into numerical feature vectors
- Captures the importance of words in the corpus

### 3. Model Training
- Train three Naive Bayes variants
- Evaluate performance on test set
- Select the best performing model

### 4. Prediction
- Preprocess input message
- Transform using TF-IDF vectorizer
- Classify using a trained model
- Return prediction with confidence score

## 📈 Model Training

To retrain the model with your own data:

1. Open `Sms_Spam_Detection.ipynb` in Jupyter Notebook
2. Load your dataset (should have 'label' and 'message' columns)
3. Run all cells to train the models
4. New `model.pkl` and `vectorizer.pkl` files will be generated

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Future Enhancements

- [ ] Add support for multilingual spam detection
- [ ] Implement deep learning models (LSTM, BERT)
- [ ] Create a REST API for integration
- [ ] Add real-time email scanning
- [ ] Develop a browser extension
- [ ] Improve UI/UX with modern frameworks

## 🐛 Known Issues

- The model may have difficulty with messages containing excessive special characters
- Performance may vary with messages in languages other than English

## 👨‍💻 Author

**Subhadip**
- GitHub: [@subhadip2510](https://github.com/subhadip2510)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the SMS Spam Collection dataset
- scikit-learn community for excellent ML tools
- Flask team for the lightweight web framework


