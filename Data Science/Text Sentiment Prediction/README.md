# If you talk to a man in a language he understands, that goes to his head. If you talk to him in his language, that goes to his heart: Text Sentiment Prediction
### Task ###
Predict sentiment for given tweets in the **sentiment140**
dataset. It contains 1,600,000 tweets, 80% for training,
20% for testing(50% for private and 50 % for public),
Extracted using the twitter api. The tweets have been
annotated (0 = negative, 4 = positive).
### Kaggle Link ###
[Kaggle](https://ppt.cc/fUz2Jx "link")

### Data Discription ###
There are 6 attributes for one instance.<br>
**sentiment**: The polarity of the tweet (0 = negative, 4 = positive)<br>
**id**       : The id of the tweet<br>
**date**     : The date of the tweet<br>
**query**	 : The query. If there is no query, then this value is
NO_QUERY.<br>
**user**	 : The username that tweeted <br>
**text**	 : The text of the tweet <br>

### Evaluation ###
It's evalutaed by **MSE(Mean Square Error)**<br>
$$\sqrt{ \frac{1}{n}\sum^{n}_{i=1}(\overline{y}_i-y_i)^2 }$$