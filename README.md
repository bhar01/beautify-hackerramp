# Submission for Myntra's hackathon - Hackerramp'21: Team Pebbles

## **Problem statement: Beautify**

Our solution is split into three major modules:

### _Foundation/ lipstick recommender_
 
Recommends products from the database based on users' skin shade, undertone, price range and other factors. Similarity/ Dissimilarity is calculated using the color distance metric and we use face detection on user input with the HaarCascade model
 
 
### _Magic mirror_

Users can try products from the recommender out on themselves using this virtual mirror.


### _Aesthetic recommender_

Recommends products from the database based on users' quiz results/input images/previous purchases. Employs FastAI's Resnet-18 and ANNOY for rcommendation
