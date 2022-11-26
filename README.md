# News title classification

In this project I digg into the headlines of IsraelHayom Hebrew website and
Haaretz Hebrew website. I will try to build machine learning model that when given a headline
from one of the two, it will classify it to it's source with high probability.
Hopefully, at list 85%.

# Data probe
First I tried to search for resources, since I wanted to get as match samples as I can.
I found out that both newspapers website offer RSS page that I can get into using http get request.
Haaretz's RSS page is formatted as needed, but IsraelHayom's RSS page has a lot of encoding errors.
It took me some time to parse their xml, but I found the pattern of their errors.
Now, the website prob function can get a normal RSS page and get the data from there.

Secondly, I wanted to get all the data from beginning of my work till handout of the script.
So I wrote a batch file that I set the Windows Task scheduler to run at each hour.
Now I have around 1,200 sample from both newspapers.
I'm saving the sample as json file with the classification (0 for Haaretz and 1 for IsraelHayom).

# Preprocessing
Since the problem is classifying text, after having some data, I have to convert it to a way that I will be able to 
apply Machine Learning algorithms.
I used sklearn CountVectorizer that set each word in the training set a number and then returns a matrix where for 
each sample in the i'th column there is 1 iff the word that got the number i is in that sample.
Then I used sklearn TfidfTransformer in order to normalize the cells.
Since I wanted to save some samples that later on I will use to evaluate real prediction, I splited the sample to 
test and train sets.
### Fetcher selection
I tried also doing some fetcher selection, but I found out that it looks like there is no fetchers to remove.
I think that is because each feacher is a word, and if I take a work from the vocabulary it has more problem when 
seeing that word in when trying to predict the test set.

#### **Note** 
that the CountVectorizer vocabulary is not fitted with the train set sample in order to get as match 
closer to real test cases, where there may be word that won't appear in the vocabulary.

# Training the model
Since the problem is classification problem I choose a few of classifications models to work with.
For each model I used cross validation method to choose its best hyperparameter over the training set.
For each hyperparameter I run the cross validation method, and then I take the mean of scores, that is in order to 
evaluate the hyperparameter in the mean case.
After choosing each model its best hyperparameter, I'm training the model again on all the training dataset and 
calculate the score of the model under the test dataset (that I saved earlier in the preprocessing stage).
In order to avoid over fitting, I choose the model that has the highest score over the test dataset and not the train 
set.
I'm saving the model using pickle library to allow fast predictions in the future without training the model all 
over again.
# Using the script
The script can be run in four configurations:
1. Update data set-Let the user update the dataset from current website RSS:
    Run the script with 'update' as second argument of the program
2. Load test - Let the user download from website a separate dataset to test the model on data that it has never seen 
   before:
    Run the script with 'load_test' as second argument of the program
3. Fit - Let the user fit the model using the ml methods mentioned above
    Run the script with 'fit' as second argument of the program
    Note: There should be dataset with sample in the directory mentioned in the top of the script (The mode need to 
   learn from something :))
4. Predict - Let the user use a trained model to clsify newspaper titles to Haaretz or IsraelHayom
    Run the script with 'predict' as second argument of the program and then the path to the data to predict third 
   argument

I had very nice time writing the script and I hope you will like it.
Thanks