# Fish Classification Using ResNet-18

The dataset contains 9 different seafood types collected from a supermarket in Izmir, Turkey
for a university-industry collaboration project at Izmir University of Economics, and this work
was published in ASYU 2020.

At this project, I used a pretrained model ResNet18. A pretrained model will make image classification work easier because of the parameters that are already trained before, except the last layer.  

At the last step, which is testing , the model obtained 99.8% accuracy. It may be a bad thing because we can assume an overfitting model from this result. But, we also need to see the data that we have. All the pictures that we used in training, validation and testing are taken at the same place and condition, almost no difference. It just showed the variety that the data has, which contribute to the high accuracy of the model at the end. 

We can aim for a better performance for real life situation only when we have more various pictures that are taken from many different conditions.  

Data source : https://www.kaggle.com/crowww/a-large-scale-fish-dataset
