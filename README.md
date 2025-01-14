# Essay-Scoring-using-NLP
Machine Learning project to predict essay scores using NLP libraries and other techniques
We used a few techniques to make a model that best predicts the the score of the essay from the textual input. We used NLP libraries such as NLTK, Bert and Spacy to extract features from the training essays. The model was then trained on those extracted features to predict the essay score. One inference was then made after tekenizing the input the same way as the training data. This approach was however constrained by the fact that the original dataset was scored by a singe human and his inference could be different from other humans or may be contradicting even to features like cohesion, grammar, sentence complexity etc. 


We used another rather simple technique which gave us better results. In this technique we used LSTM to predict the scores of the essays. The textual data was simply input to the model and it was able to perform well when used to check for results.
