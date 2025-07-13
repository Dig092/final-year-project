# Task
Extract the part of the tweet that reflects the sentiment (positive, negative, or neutral) given by the provided sentiment label.

# Evaluation Metric
The evaluation metric for the competition is the word-level Jaccard score.

# Submission Format
- A CSV file with the following columns:
  - `textID`: Unique ID for the tweet
  - `selected_text`: The portion of the tweet that reflects the sentiment

Example:
textID,selected_text
f941f4d7fa,"what a great day"
f941f4d7fa,"it was ok"


# Dataset
The dataset consists of tweets along with sentiment labels.

### Dataset Structure:
- **twitter_training.csv**: Contains the training dataset with text, sentiment labels, and selected text.
- **twitter_validation.csv**: Contains the test dataset with text and sentiment labels.
- **sample_submission.csv**: A sample format of the submission file.

### Columns in the Dataset:
- **textID**: Unique identifier for each tweet.
- **text**: The tweet itself.
- **sentiment**: Sentiment label of the tweet, which can be positive, negative, or neutral.
- **selected_text**: (in train.csv only) The portion of the tweet that reflects the sentiment.

### Dataset Name:
- Dataset can be download from kaggle using `jp797498e/twitter-entity-sentiment-analysis`

Assume runtime already has kaggle API Auth configured.

Top score is 0.74 in score try to beat it!
