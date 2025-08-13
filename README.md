📩 SMS Spam Detection using Naive Bayes
📌 Project Overview
This project implements a Spam Detection System for SMS messages using three variants of the Naive Bayes algorithm:

GaussianNB

MultinomialNB

BernoulliNB

The goal is to classify messages as either Spam or Ham (Not Spam) using probabilistic learning based on Bayes’ theorem.

🛠 Features
Text preprocessing (cleaning, tokenization, vectorization)

Implementation of three Naive Bayes models for comparison

Performance evaluation using accuracy, precision, recall, and F1-score

Simple and interpretable workflow

📂 Dataset
We used the SMS Spam Collection Dataset, which contains SMS messages labeled as "spam" or "ham".

Source: UCI Machine Learning Repository

Total messages: ~5,572

🧠 Models Used
Model	Description
GaussianNB	assumes features follow a normal distribution
MultinomialNB	is Best for discrete counts such as word frequencies
BernoulliNB	Suitable for binary/boolean features

📊 Model Performance
Model	Accuracy	Precision	Recall	F1-score
GaussianNB	0.86	0.65	0.78	0.71
MultinomialNB	0.98	0.96	0.94	0.95
BernoulliNB	0.97	0.95	0.92	0.93

✅ Best model: MultinomialNB – offers the best balance of precision and recall for spam detection.
