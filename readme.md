# ARMedia - Alphabet Recognition with MediaPipe using JavaScript

This project aims to do ASL alphabet recognition real time using MediaPipe Hands's API in JavaScript.

## Summary

It is trained a Keras model that approximates a Support Vector Machine (dataset used: https://www.kaggle.com/signnteam/asl-sign-language-pictures-minus-j-z). The Keras model obtained is converted to a TensorFlow.js model and imported in a script that performs alphabet recognition using the input of the webcam. The application realized simply runs on the browser (also from mobile) without installation.

## Setup

To run a local setup of the project install *http-server* with 

```
npm install --global http-server
```

and in the folder *model* launch

```
http-server --cors
```

Inside *index.html* set the following line of code: 

```
model = await tf.loadGraphModel('http://localhost:8080/model.json');
```

while if you want to deploy the application on a server, point to the folder that contains the model:

```
model = await tf.loadGraphModel('./model/model.json');
```

## Example of use
![Y](https://user-images.githubusercontent.com/52381926/108563898-8d23e200-7302-11eb-94a5-2c15faca0273.png)
