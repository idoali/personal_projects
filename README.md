# Project Descriptions :notebook_with_decorative_cover:	

## Developing French-English Translator Using Transformer From PyTorch and TorchText 
_(first uploaded on December 2021)_

**Description**  

French/English parallel texts for training translation models. Over 22.5 million sentences in French and English. Dataset created by Chris Callison-Burch, who crawled millions of web pages and then used a set of simple heuristics to transform French URLs into English URLs, and assumed that these documents are translations of each other. This is the main dataset of Workshop on Statistical Machine Translation (WML) 2015 Dataset that can be used for Machine Translation and Language Models. For more information, please go to this paper http://www.statmt.org/wmt15/pdf/WMT01.pdf

**Objective**  

Creating a decent model with its main purpose is to translate french text to english text using normal computing resources.
Solutions : This model was developed using PyTorch and its NLP libraries TorchText. The main focus of this model is its Transformer layer, which is a state-of-art deep learning model in NLP. 

**Conclusion**  

We developed a translation model using Transformer from TorchText because they already have a built-in function for that. The biggest hurdle for this project is a limited computational power. In order for a model like this to have a decent performance, computational power is a huge factor. 

**Data source**  

https://www.kaggle.com/dhruvildave/en-fr-translation-dataset

## Creating Fish Classification Using ResNet-18 From PyTorch 
_(First uploaded on October 2021)_

**Introduction**

This dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey for a university-industry collaboration project at Izmir University of Economics, and this work was published in ASYU 2020. The dataset includes gilt head bream, red sea bream, sea bass, red mullet, horse mackerel, black sea sprat, striped red mullet, trout, shrimp image samples.

**Objective** 

Making an image classification that specifies on detecting fish. The aim is to make a model that has high performance and is trained in a short time.

**Solutions** 

Using a pre-trained model is one of the best ways to create an image classification with high performance in a short training time.

**Conclusion**

At testing , the model obtained 99.8% accuracy. It may be a bad thing because we can assume an overfitting model from this result. But, we also need to see the data that we have. All the pictures that we used in training, validation and testing are taken at the same place and condition, almost no difference. It just showed the variety that the data has, which contributes to the high accuracy of the model at the end.
We can aim for a better performance for real life situations only when we have more various pictures that are taken from many different conditions.

**Original Data Source** 

https://www.kaggle.com/crowww/a-large-scale-fish-dataset

## Amazon Phone Review Analysis & Summarization Using T5 Model 
_(First Uploaded on September 2021)_

**Description** 

This data is about phone reviews from Amazon customers. In this dataset, we can find how satisfied customers are for a certain product that they bought and their thoughts in detail. The brands that their data are served in this dataset are:
1. Motorola
2. Samsung
3. Apple
4. Sony
5. Nokia
6. Huawei
7. Google
8. ASUS
9. OnePlus
10. Xiaomi
This data was started to be taken in 2005.

**Objective** 

Analyze the review of the phones that are purchased on Amazon. These reviews can also be useful for the service in the future, so extracting information from these reviews is also important.  

**Solutions** 

In Exploratory Data Analysis (EDA), we explored the total sales and rating of the most popular phones sold on Amazon. This analysis gave us useful insight in how the phone market works for the past few years. We extracted the reviews of selected products and summarized them using T5, a popular Transformer model mostly used for summarization.  
Conclusion : Chinese phone products have a high possibility of getting high ratings. OnePlus is one of these products, and it’s the most expensive one. T5 model on this project can be executed and give a good performance. This is one of the summarization result for Samsung Galaxy Note 9 that was sold in 2019 : 

_“Clearly has visible scratch from further than 12 inches so already fails Amazon Renewed specifications . Likely to return this unless it functions very well and the scratch on the screen doesn't reflect too much to be noticed throughout the day . Likely going to return it unless it works very well . Item came scratched. Item was scratched. Returning. Item came scratching. Item went missing. Item returned. Item is scratched. Items are scratched. It's scratched. We're scratching. We'll get it back. Item will be scratched. Back to Mail Online home . Back to the page you came from . Phone would not recognize my Verizon sim card, ( I did order a Verizon phone) it imei # was unavailable . Phone would restart over and over. I did factory reset still not working. I will be sending phone back to Verizon . Will be sending it back . Screen protector on phone arrived with a screen protector that didn't fit, wasn't attached properly, and was hiding a significant number of scratches . Spent significant amount of time on the phone with Samsung support, who have advised that the screen is clearly defective . Now I have to drive 45 minutes into town to find a UPS store to return this item, and then wait for them to refund my money . “_

**Original Data Source**

https://www.kaggle.com/grikomsn/amazon-cell-phones-reviews

## Analysis on Melbourne House Market During Its Bubble Time in 2016-2018
_(First uploaded on September 2021)_

**Description**

Melbourne was experiencing a housing bubble around 2016 – 2018 (some experts say it might even burst). As we all know, the bubble is when an item sold in the market is overvalued. This can cause a burst in the future, which is a huge drop in price while the demand of the item is still low.

**Objective** 

Find how this market really works. Give the best advice for anyone who wants to purchase a house during this time.

**Solutions**
Analyzed the market through Exploratory Data Analysis (EDA) and Feature Engineering to see how each variable affects one another. 

**Conclusions** 

1. The most important factor that contributes to the price of a house is its distance to the city center and its building area.
2. The price of the house always declines around January and February.
3. If you are looking for a cheap house located close to Melbourne City Business District, Western Metropolitan is the cheapest region. But if you are really looking for the cheapest house in Melbourne and don't really care about its distance to the CBD, then Western Victoria is the place for you my friend.

**Original Data Source**

https://www.kaggle.com/anthonypino/melbourne-housing-market

## Analysis on Job Satisfaction of Employees 
_(First uploaded on June 2021)_

**Description**

A large company named XYZ, employs, at any given point of time, around 4000 employees. However, every year, around 15% of its employees leave the company and need to be replaced with the talent pool available in the job market. The management believes that this level of attrition (employees leaving, either on their own or because they got fired) is bad for the company

**Objective**

Find what factors the company should focus on, in order to curb attrition. In other words, they want to know what changes they should make to their workplace, in order to get most of their employees to stay. 

**Solutions**

Analyzed the data through Exploratory Data Analysis (EDA) and Feature Engineering to see how each variable affects one another. 

**Conclusions** 

Number companies worked The number one feature, which is number companies worked probably means a number an employee has worked for before.
Percent salary hike which means the increase of salary that an employee has. From this, we know that what determines employee's satisfaction is the increase of salary rather than the number of salary itself. It proved how emotional a human being is.
Stock options level This is how much stock they have on the company. Some high positions in a company let that employee have some stocks in the company.

**Original Data Source**

https://www.kaggle.com/vjchoudhary7/hr-analytics-case-study

## Analysis & Modeling NBA Players Salaries By Their Physical Attributes Using Various Machine Learning Algorithms 
_(First uploaded on June 2021)_

**Introduction**

This data contains the information of NBA players, some of their physical attributes and their salary. We wanted to know what attributes affect the NBA players salaries. Also, it would be fun if we could make a machine learning model that can predict an NBA player’s salary, so we tried to make that out too.

**Objective**

Analyze various attributes of  a player that can contribute to their salaries as a player. At the end, build an effective model to predict the salary of a player with certain attributes.

**Solutions**

Analysis done by doing feature engineering and EDA (Exploratory Data Analysis). We made models using SVM (Support Vector Machine), Random Forest, Ada-Boost, and Neural Network then compared their performance. The model with the best accuracy would be chosen.

**Conclusions** 

Certain attributes contribute to a player’s salary, but mostly it is affected by their whole performance. Based on this project, Random Forest is the best algorithm to predict NBA players salary, compared to one-layered Neural Network, SVM and Ada-Boost.  

**Original Data Source** 

https://www.kaggle.com/isaienkov/nba2k20-player-dataset


## Analysis & Modeling On US Citizen’s Wages Data Using Various Machine Learning Algorithms 
_(First uploaded on June 2021)_

**Introduction** 

The data has a salary column along with the other categories to predict the income of the people based on a demographic. Those categories that may affect salary and also are in this data are : 
1. Age
2. Work class
3. Education
4. Marital-status
5. Occupation
6. Relationship
7. Race
8. Sex
9. Capital-gain
10. Capital-loss
11. Hours-per-week
12. Native-country

**Objective**

Analyze various attributes from the data that can contribute to a citizen’s salary. At the end, build an effective model chosen from various models, to predict whether a citizen has a salary more than US$50,000 a year or not.

**Solutions** 

Understanding the data by doing Feature Engineering and EDA (Exploratory Data Analysis).  Comparing the performance of Linear Regression, Logistic Regression, Gaussian Naive-Bayes, K-Neighbors Classifier, Support Vector Classifier, Decision Tree, AdaBoost Classifier, and Random Forest with their default hyperparameters. The best model would be fine-tuned using Grid Search CV to find the best hyperparameters. 

**Conclusions** 

There are many factors contributing to how much wage a worker will get, from education to experience
Jobs with the most workers to be paid more than 50 thousands dollars are Executive Managerial and Professional.
The best prediction model we can find is Ada Boost Classifier with the parameters: (n_estimator = 120, learning_rate = 1).

**Original Data Source**

https://www.kaggle.com/ddmasterdon/income-adult

## Finding Football Best Talents Using Exploratory Data Analysis 
_(First uploaded on June 2021)_

**Introduction**

Detailed attributes for every player registered in the latest edition of FIFA 19 database. This dataset can be used for many things because it has a lot of information about a player's physical and technical ability. 

**Objective**

Finding best young talents to be recruited. Their price and salary play a big role in this.
  
**Solutions**

Analyzed all their physical and technical attributes based on the role they played in the field. At the end, we compared the results from each player to find the best.  

**Conclusions**

We found 60 best young talents in 6 different positions. We make sure that they have affordable prices so most clubs in Europe can still purchase them. 
 
**Original Data Source**

https://www.kaggle.com/karangadiya/fifa19

## Exploratory Data Analysis on UK Car Market and Car Price Prediction Using Deep Learning 
_(First uploaded on June 2021)_

**Introduction**

This cleaned dataset contains information of price, transmission, mileage, fuel type, road tax, miles per gallon (mpg), and engine size of over 10 thousands cars sold in United Kingdom. This dataset can be useful to gain insight on the car market. 

**Objective**

Analyze the car market in United Kingdom and gain as many insights as possible. At the end, build a Machine Learning model to predict the sale price of a car.

**Solutions**

We did Exploratory Data Analysis on the dataset to gain insight on the market. After that, we created a Deep Learning model to predict the price of a car sold in United Kingdom.

**Conclusions**

Different specifications of a car give different prices. But one of the most important factors is their brand. The model that we developed using Deep Learning gave a good performance with low error on predicting a car’s price.

**Original Data Source**

https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes


## Tweets Sentiment Analysis 
_(First uploaded on June 2021)_

**Introduction**

Identifying emotions has become an integral part of many NLP and data science projects. With the help of this dataset, one can train and build various robust models and perform emotional analysis. Manual annotation of the dataset to obtain real-valued scores was done through Best-Worst Scaling (BWS), an annotation scheme shown to obtain very reliable scores (Kiritchenko and Mohammad, 2016). The data is then split into a training set and a test set.

**Objective**

There are four emotion classifications on this dataset, which are anger, fear, sad and joy. Make the best model that can classify a tweet based on these emotions. 

**Solutions**

We compared the performance of Word Embeddings and Countvectorizer to map each word in the tweets to numerical features. Then, we built a good performance Machine Learning model by comparing some of the most popular Machine Learning algorithms.   

**Conclusions**

From this project, we find that CountVectorizer is better than Word Embeddings to map words to numerical features or vectors at least in this case. The classification is done by using Logistic Regression, because it has the highest accuracy among the models we tested.

**Original Data Source**

https://www.kaggle.com/anjaneyatripathi/emotion-classification-nlp


## Analysis On Factors That Can Contribute to Stroke & Building A Model to Predict A Stroke Patient 
_(First uploaded on May 2021)_

**Introduction**

According to the World Health Organization (WHO) stroke is the 2nd leading cause of death globally, responsible for approximately 11% of total deaths.
This dataset is used to predict whether a patient is likely to get a stroke based on the input parameters like gender, age, various diseases, and smoking status. Each row in the data provides relevant information about the patient.

**Objective**

Analyze the data to find insight on what factors contribute to stroke. Find a machine Learning model that can predict whether someone has a stroke or not. 

**Solutions**

We compared each attribute on the data to find what really contributes to stroke. At the end, We compared many Machine Learning Algorithms on their performance predicting a patient for having a stroke or not. 

**Conclusions**

This stroke problem that we have here is a binary classification problem. For this case alone, the best models that we can use are SVM for classification and Logistic Regression with a score of 94.6%. The models that we would avoid to use are Naive-Bayes and Linear Regression. This is because of a very poor result and performance from this model.

**Original Data Source**

https://www.kaggle.com/fedesoriano/stroke-prediction-dataset

## Comparing Ensemble Methods to Predict Future Sales On Russian Software Market 
_(First uploaded on March 2021)_

**Introduction**

This challenge serves as the final project for the "How to win a data science competition" Coursera course. A disclaimer, I never took this course. I got this dataset from Kaggle. This is a dataset kindly provided by one of the largest Russian software firms - 1C Company. 

**Objective**

Analyze the current software market in Russia and find the insight on the market.. At the end, Build a Machine Learning model that can predict how the market works in the future. 

**Solutions**

Analysis on the market is done by doing EDA (Exploratory Data Analysis). From this step also, we obtained insights on the current software market in Russia. At the end, we compared various ensemble methods to find which model works best. 

**Conclusions**

Both ensemble methods that we compared on this project, which are AdaBoost and Random Forest worked equally well. 

**Original Data Source** 

https://www.kaggle.com/c/competitive-data-science-predict-future-sales

## A Study On Various Factors Contributed to Borrower's Ability to Pay Debts 
_(First uploaded on January 2020)_

**Introduction**

This is the data from an American credit card company. This dataset contained various information about credit card users of this company. If this dataset is analyzed correctly, it can help the company to acknowledge the pattern of a good credit card user.  

**Objective**

Analyze every possible factor that may affect the ability of borrowers to repay back their loaned money.

**Solutions**

We compared each feature from the dataset with the user’s ability to pay their loans using statistical analysis and graph comparisons. 

**Conclusions**

1. With sentiment analysis, surprisingly borrowers with more subjective and negative reasons have better ability to pay their debts. Borrowers who have their own house also have better ability. There is no correlation of the amount of loan with borrower's annual income or DTI.
2. "Renewable energy" and "Educational" are two major reasons with the highest probability of borrowers not paying debts. 
3. Iowa is the state with lowest loan payback ability by ratio 30 %. 
4. High interest rates can trigger low loan payback ability. 
_I am very open to any suggestion. It can be on the method i analyzed the data, or even the way i understand important terms used._

## Investigating the relationship between the playing surface and the injury and performance of NFL athletes?
_(First uploaded on January 2020)_
**Introduction**

These are datasets and variables provided to examine the effects that playing on synthetic turf versus natural turf can have on player movements and the factors that may contribute to lower extremity injuries. The data provided for analysis are 250 complete player in-game histories from two subsequent NFL regular seasons. Three different files in .csv format are provided, documenting injuries, player-plays, and player movement during plays. This manual describes the specifics of each variable contained within the datasets as well as guidelines on the best approach to processing the information.

**Objective**

The task is to investigate the relationship between the playing surface and the injury and performance of National Football League (NFL) athletes and to examine factors that may contribute to lower extremity injuries.

**Solutions**

The task was solved by doing Exploratory Data Analysis. We found patterns between positions, time spent in the field and what injuries that the players have. For different positions, the players are prone to different kinds of injuries.

**Original Data Source**

https://www.kaggle.com/c/nfl-playing-surface-analytics/overview

## Training A Deep Learning Model to Classify Cats & Dogs Pictures
_(First uploaded on December 2019)_

**Introduction**

In the era in which AI is becoming more common, there is one area where AI can be useful. That is the Image classifier. We always see this time and time again, that AI’s ability in this area can be very useful in automation. In this project, the task was to build an image classifier that can perform well to classify whether a picture is a picture of a cat or a picture of a dog.  

**Objective**

Building an image classifier to classify a picture to be a picture of a dog or a cat. 

**Solutions**

We chose a deep learning model to be the classifier. The type of deep learning that we chose is CNN (Convolutional Neural Network) which is the most common deep learning model to be used for image classification. 

**Original Data Source**

https://www.kaggle.com/tongpython/cat-and-dog

## Choosing the Best Machine Learning Algorithm to Predict A Pulsar
_(First uploaded on December 2019)_

**Introduction**

Pulsars are a rare type of Neutron star that produce radio emission detectable here on Earth. They are of considerable scientific interest as probes of space-time, the interstellar medium, and states of matter. Machine learning tools are now being used to automatically label pulsar candidates to facilitate rapid analysis. Classification systems in particular are being widely adopted,which treat the candidate data sets as binary classification problems.

Each candidate is described by 8 continuous variables, and a single class variable. The first four are simple statistics obtained from the integrated pulse profile (folded profile). This is an array of continuous variables that describe a longitude-resolved version of the signal that has been averaged in both time and frequency . The remaining four variables are similarly obtained from the DM-SNR curve . These are summarized below:

1. Mean of the integrated profile.
2. Standard deviation of the integrated profile.
3. Excess kurtosis of the integrated profile.
4. Skewness of the integrated profile.
5. Mean of the DM-SNR curve.
6. Standard deviation of the DM-SNR curve.
7. Excess kurtosis of the DM-SNR curve.
8. Skewness of the DM-SNR curve.
9. Class

**Objective**

Compare various Machine Learning algorithms to find which algorithm works best to predict a Pulsar.

**Solutions**

Scikit-Learn and Keras are used during this project to build the machine learning model. The accuracy from each model is later compared.

**Original Data Source**

https://www.kaggle.com/colearninglounge/predicting-pulsar-starintermediate?select=pulsar_data_train.csv

## Kepler Exoplanet
_(First uploaded on December 2019)_

**Introduction**

The Kepler Space Observatory is a NASA-build satellite that was launched in 2009. The telescope is dedicated to searching for exoplanets in star systems besides our own, with the ultimate goal of possibly finding other habitable planets besides our own. The original mission ended in 2013 due to mechanical failures, but the telescope has nevertheless been functional since 2014 on a "K2" extended mission.

Kepler had verified 1284 new exoplanets as of May 2016. As of October 2017 there are over 3000 confirmed exoplanets total (using all detection methods, including ground-based ones). The telescope is still active and continues to collect new data on its extended mission.

This dataset is a cumulative record of all observed Kepler "objects of interest" — basically, all of the approximately 10,000 exoplanet candidates Kepler has taken observations on. This dataset has an extensive data dictionary, which can be accessed here. Highlightable columns of note are:

**kepoi_name**: A KOI is a target identified by the Kepler Project that displays at least one transit-like sequence within Kepler time-series photometry that appears to be of astrophysical origin and initially consistent with a planetary transit hypothesis
**kepler_name**: [These names] are intended to clearly indicate a class of objects that have been confirmed or validated as planets—a step up from the planet candidate designation.
**koi_disposition**: The disposition in the literature towards this exoplanet candidate. One of CANDIDATE, FALSE POSITIVE, NOT DISPOSITIONED or CONFIRMED.
**koi_pdisposition**: The disposition Kepler data analysis has towards this exoplanet candidate. One of FALSE POSITIVE, NOT DISPOSITIONED, and CANDIDATE.
**koi_score**: A value between 0 and 1 that indicates the confidence in the KOI disposition. For CANDIDATEs, a higher value indicates more confidence in its disposition, while for FALSE POSITIVEs, a higher value indicates less confidence in that disposition.

**Objective**

Create a machine learning model that can perform well to predict an exoplanet. 

**Solutions**

There are various machine learning models that we compared such as Artificial Neural Network, Support Vector Machine, Random Forest Classifier etc. The model with the highest accuracy will be chosen to be our model.

**Original Data Source**

https://www.kaggle.com/nasa/kepler-exoplanet-search-results

