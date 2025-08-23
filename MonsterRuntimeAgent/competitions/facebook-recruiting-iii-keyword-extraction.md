
# Task

The task is to predict tags (a.k.a. keywords, topics, summaries), given only Stack Exchange question text and its title.

# Evaluation Metric

Mean F1-Score. Top score is 0.81350. We need to beat it.

# Submission Format

For every question in the test set, your submission file should contain two columns: Id and Tags. Id is the unique identifier of a question in test.csv. Tags should be a space-delimited list predicted tags. You should maintain the order of questions.

The file should contain a header and have the following format:

```
Id,Tags
1,"c++ javaScript"
2,"php python mysql"
3,"django"
etc.
```

# Dataset 

**Train.csv** contains 4 columns: Id,Title,Body,Tags

- Id - Unique identifier for each question
- Title - The question's title
- Body - The body of the question
- Tags - The tags associated with the question (all lowercase, should not contain tabs '\t' or ampersands '&')

**Test.csv** contains the same columns but without the Tags, which you are to predict.

Dataset can be downloaded using Kaggle CLI with this command:
```bash
kaggle competitions download -c facebook-recruiting-iii-keyword-extraction
```

Use GPU if available.