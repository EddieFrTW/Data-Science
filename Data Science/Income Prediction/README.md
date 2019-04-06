# Money Brings Satisfaction But Not Happiness: Income Prediction
### Task ###
Predict whether income exceeds $50K/yr based on
census data.<br>
It is a binary classification problem.<br>
※ Note that in the train and test data, **salary > 50K**
represents as **1** and **salary ≦ 50K** represents as **0**
### Kaggle Link ###
[Kaggle](https://reurl.cc/VYROZ "link")

### Data Discription ###

14 attributes for one instance.<br>
**Age**<br>
**Workclass**<br>
**fnlwgt**: The number of people the census takers believe that observation represents.<br>
**Education** <br>
**Education-num**<br>
**Marital-status**<br>
**Occupation**<br>
**Relationship**: Wife, Own-child, Husband, Not-in-family, Otherrelative,
Unmarried.<br>
**Race**<br>
**Sex** <br>
**Capital-gain**<br>
**Capital-loss**<br>
**Hours-per-week**<br>
**Native-country**<br>
### Evaluation ###
It's evalutaed by **F1 score**<br>
 $$F1 \ score = 2\frac{precision*recall}{precision+recall}$$, where<br>
 $$precision = \frac{True \ Positive}{True \ Positive+False \ Positive}$$ 
 $$recall = \frac{True \ Positive}{True \ Positive+False \ Negative}$$ 

