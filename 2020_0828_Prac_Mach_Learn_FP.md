---
title: "2020_0828_Prac_Mach_Learn_FP"
author: "John Nguyen"
date: "8/28/2020"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
library(lattice)
library(ggplot2)
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(tibble)
library(bitops)
library(rattle)
library(randomForest)
```

### Include the libraries and then clean the Data

To complete this project we need to include specific libaries.

Then we need to get rid of spurious variables. Any variable that include the words, such as kurtosis, Max, Min, var, stddev, avg,skewness, timestamp, raw, name, and window can be removed because they will not help with prediction modeling. We further clean the data with nearzerovar function. This removes variables with approximately 0 variance.

```{r cars, echo=TRUE}
library(lattice)
library(ggplot2)
library(caret)
library(dplyr)
library(rpart)
library(rpart.plot)
library(tibble)
library(bitops)
library(rattle)
library(randomForest)

train <- read.csv(file = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", header=TRUE)
final_test<-read.csv(file = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", header=TRUE)
train<-data.frame(train)

new_train<-train%>%
    select(-contains('kurtosis'))%>%
    select(-contains('skewness'))%>%
    select(-contains('stddev'))%>%
    select(-contains('max'))%>%
    select(-contains('min'))%>%
    select(-contains('var'))%>%
    select(-contains('avg'))%>%
    select(-contains('raw'))%>%
    select(-contains('timestamp'))%>%
    select(-contains('name'))%>%
    select(-contains('window'))
new_train<- new_train[, colSums(is.na(new_train)) == 0] 
nearZvar <- nearZeroVar(new_train)
new_train<-new_train[,-nearZvar]
new_train<-new_train[,-1]

inValidation = createDataPartition(new_train$classe, p = 3/4)[[1]]
training = new_train[ inValidation,]
validation = new_train[-inValidation,]
```

### Tree Method
```{r Tree, echo=TRUE}
set.seed(11122)
Tree_mod <- rpart(classe ~ ., data=training, method="class")
fancyRpartPlot(Tree_mod)
predictTreeVal<-predict(Tree_mod, newdata=validation, type="class")
testclassvalid<-as.factor(validation$classe)
confusionMatrix(predictTreeVal,factor(validation$classe))
out_sample_error_tree<-1-confusionMatrix(predictTreeVal,factor(validation$classe))$overall[[1]]
out_sample_error_tree
```

### Linear Discriminant Analysis
```{r LDA, echo=TRUE}
set.seed(87234)
modelda  <- train(classe ~ ., data=training, method = "lda")
predictlda <- predict(modelda, newdata=validation)
confusionMatrix(predictlda,factor(validation$classe))
out_sample_error_lda<-1-confusionMatrix(predictlda,factor(validation$classe))$overall[[1]]
out_sample_error_lda
```

### Random Forest
```{r RandomForest, echo=TRUE}
set.seed(98427)
controlfunction <- trainControl(method="cv", number=3, verboseIter = FALSE)
rand_for <- train(classe ~ ., data=training, method="rf", trControl=controlfunction)
predrf<-predict(rand_for, newdata=validation)
confusionMatrix(predict(rand_for, newdata=validation), factor(validation$classe))
out_sample_error_rf<-1-confusionMatrix(predict(rand_for, newdata=validation), factor(validation$classe))$overall[[1]]
out_sample_error_rf
```
Random Forest has the highest accuracy and lowest out of sample error of the 3 models. Linear Discriminant Analysis has lowest accuracy and the highest out of sample error out of the 3 models. The reason for Random Forest to have a high accuracy for is maybe do to over fitting.

```{r RandomForest2, echo=TRUE}
plot(rand_for)
```

### Go with Random Forest for Final Prediction
```{r Final_Model, echo=TRUE}
set.seed(80275)
pred_model<-predict(rand_for, newdata=final_test)
pred_model
```

