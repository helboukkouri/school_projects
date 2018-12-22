# Naive Bayes for Text Classification

When working with text data, we are often faced with a **large number of features**. In fact, Bag of Words - which is the usual method by which text is transformed into **numerical features** - consists of transforming each text into a list of **word occurences** for an often very large set of words called the vocabulary. Thankfully, there are algorithms that can cope with such a high number of features : **Naive Bayes classifiers**.

To illustrate the ability of Naive Bayes to perform on a large set of features, we will use it on the **Sentiment Analysis** task associated with **IMDB Movie Reviews dataset**.
