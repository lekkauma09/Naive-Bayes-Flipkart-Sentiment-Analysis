# Naive Bayes Flipkart Sentiment Analysis

## Project Description

This project performs sentiment classification on Flipkart product reviews using the Naïve Bayes algorithm.

## Dataset

Dataset: Dataset-SA.csv

Columns:
- product_name
- product_price
- Rate
- Review
- Summary
- Sentiment

## Libraries Used

- NumPy
- Pandas
- Scikit-learn

## Algorithm

1. Load the dataset.
2. Remove missing values.
3. Combine Review and Summary.
4. Convert text into numerical features using CountVectorizer.
5. Split the dataset into training and testing sets.
6. Train the Multinomial Naïve Bayes model.
7. Predict the sentiment.
8. Calculate accuracy.

## Output

The model predicts:

- Positive
- Negative
- Neutral

## Author

Lekka Uma
