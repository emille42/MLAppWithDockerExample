This is docker container app for deploy simple logistic regression model trained on heart cleveland dataset:
https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

How to run this app?

Execute next commands:
1. docker build -t mlcl .
2. docker run -d --name mlcl -p 80:80 mlcl
3. Follow link : http://127.0.0.1/docs