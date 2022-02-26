spam.df <- readRDS("spam_base_v2.RData")

drop.list  <- c('u','train','test');
spam.train <- subset(spam.df,train==1)[, !(names(spam.df) %in% drop.list)];
spam.test  <- subset(spam.df,test==1)[, !(names(spam.df) %in% drop.list)];

dim(spam.train)
dim(spam.test)
head(spam.train)
str(spam.train)
library(tidyverse)
library(pROC)
library(xgboost)
library(MASS)
library(woeBinning)
library(OneR)

#install.packages('naivebayes',dependencies=TRUE)
library(naivebayes)

train.matrix <- as.matrix(spam.train[,-58])

xg %>% 0 <- xgboost(data=train.matrix, label=spam.train$spam, max_depth=4, nrounds=10, objective='binary:logistic');

# Plot variable importance;
importance.10 <- xgb.importance(feature_names=colnames(train.matrix),model=xg.10)


# Plot all variables;
xgb.plot.importance(importance.10, rel_to_first=TRUE, xlab='Relative Importance',
                    main='XGBoost Model')
# Where is the OneR winner?


# Plot top 10 variables;
xgb.plot.importance(importance.10[1:10], rel_to_first=TRUE, xlab='Relative Importance',
                    main='XGBoost Model')


feature.list <- importance.10$Feature[1:10]


##########################################
# char_freq_dollar
##########################################
woe.01 <- woe.binning(df=spam.train,target.var=c('spam'),pred.var=c('char_freq_dollar'))

# WOE plot for age bins;
woe.binning.plot(woe.01)

# Score bins on data frame;
woe.df <- woe.binning.deploy(df=spam.train,binning=woe.01)
woe.test <- woe.binning.deploy(df=spam.test,binning=woe.01)



for (feat in feature.list) {
  woe <- woe.binning(df=spam.train,target.var=c('spam'),pred.var=c(feat))
  print(woe)
  
}


feature.list

model.1 <- glm(spam ~ word_freq_remove + 
                capital_run_length_total + capital_run_length_longest +
                word_freq_edu + char_freq_xpoint + 
                word_freq_hp + word_freq_free + word_freq_george + 
                word_freq_our,
                data=spam.train, family='binomial')
summary(model.1)


train <- sample(nrow(gc), round(0.6*nrow(gc)))

file.name <- 'model1.html';
stargazer(model.1, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Model #1: Logistic Regression'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE)

model.score.1 <- model.1$fitted.values
model.score.1 <- predict()
roc.1 <- roc(response=spam.train$spam, predictor=model.score.1)
print(roc.1)
plot(roc.1)

auc.1 <- auc(roc.1)


coords(roc=roc.1,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.1,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.train$ModelScores <- model.1$fitted.values
spam.train$classes <- ifelse(spam.train$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.train$spam, spam.train$classes)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$classes);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r


stargazer(t/r, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Model #1: Confusion Matrix'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE)



model.score.1.test <- predict(model.1, spam.test)
roc.1.test <- roc(response=spam.test$spam, predictor=model.score.1)
print(roc.1.test)
plot(roc.1)

auc.1 <- auc(roc.1)


coords(roc=roc.1.test,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.1.test,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.test$ModelScores <- predict(model.1, spam.test)
spam.test$classes <- ifelse(spam.test$ModelScores>roc.specs$threshold,1,0)
spam.test$classes <- ifelse(spam.test$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.test$spam, spam.test$classes)

# Rough confusion matrix using counts;
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.test$spam, spam.test$classes)
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r

################
## WOE + logit regression 
filterlist <- c(feature.list,'spam')
reducedDF <- spam.train %>% 
  select(filterlist)

names(reducedDF) == filterlist
dataset.woe <- woe.binning.deploy(reducedDF, woe)


model.score.2 <- model.2$fitted.values
roc.2 <- roc(response=spam.train$spam, predictor=model.score.2)
print(roc.2)
plot(roc.2)

auc.2 <- auc(roc.2)


coords(roc=roc.2,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.2,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.train$ModelScores <- model.2$fitted.values
spam.train$classes <- ifelse(spam.train$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.train$spam, spam.train$classes)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$classes);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r

############test###########
reducedDF <- spam.test %>% 
  select(filterlist,'spam')

woe <- woe.binning(df=reducedDF,target.var=c('spam'),pred.var=feature.list)

dataset.woe <- woe.binning.deploy(reducedDF, woe)

model.score.2.test <- predict(model.2, dataset.woe)


roc.2.test <- roc(response=spam.test$spam, predictor=model.score.2)
print(roc.2.test)
plot(roc.2)

auc.2 <- auc(roc.2)


coords(roc=roc.2.test,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.2.test,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.test$ModelScores <- predict(model.2, dataset.woe)
spam.test$classes <- ifelse(spam.test$ModelScores>roc.specs$threshold,1,0)
spam.test$classes <- ifelse(spam.test$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.test$spam, spam.test$classes)

# Rough confusion matrix using counts;
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.test$spam, spam.test$classes)
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r

###########
#naive Bayes
library(naivebayes)
nb.1 <- naive_bayes(y = as.factor(dataset.woe$spam), x= dataset.woe %>% select(-'spam'))
names(dataset.woe)
# Compare the table to the table above;
plot(nb.1)

predicted.class <- predict(nb.1)
pct.accuracy <- mean(predicted.class==dataset.woe$spam)

predicted.classProb <- predict(nb.1,type=c('prob'))

prob.B <- mean(predicted.class==dataset.woe$spam)
prob.NB <- 1-mean(predicted.class==dataset.woe$spam)


spam.train$predicted.classNB <- predicted.class

df.1 <- as.data.frame(list(SpamRisk=dataset.woe$spam,Class=dataset.woe %>% select(-'spam')))

spam.train$ModelScoresNB <- predicted.class$fitted.values
spam.train$classesNB <- ifelse(spam.train$ModelScoresNB>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.train$spam, spam.train$predicted.classNB)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$predicted.classNB);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r
######################test##
predicted.class <- predict(nb.1,spam.test)
pct.accuracy <- mean(predicted.class==dataset.woe$spam)

predicted.classProb <- predict(nb.1,type=c('prob'))

prob.B <- mean(predicted.class==dataset.woe$spam)
prob.NB <- 1-mean(predicted.class==dataset.woe$spam)


spam.test$predicted.classNB <- predicted.class

df.1 <- as.data.frame(list(SpamRisk=dataset.woe$spam,Class=dataset.woe %>% select(-'spam')))

spam.test$ModelScoresNB <- predicted.class$fitted.values
spam.test$classesNB <- ifelse(spam.test$ModelScoresNB>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.test$spam, spam.test$predicted.classNB)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$predicted.classNB);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r

################################
reducedDF <- spam.train %>% 
  select(filterlist)

binning  <- woe.binning(reducedDF,'spam',reducedDF)
woe.binning.plot(binning)
tabulate.binning <- woe.binning.table(binning) 
tabulate.binning
df.with.binned.vars.added <- woe.binning.deploy(reducedDF, binning,
                                                add.woe.or.dum.var='woe')
df.with.binned.vars.added <- df.with.binned.vars.added %>% 
  select(-filterlist)
names(df.with.binned.vars.added)
df.with.binned.vars.added$spam <- spam.train$spam

model.2 <- glm(spam ~ .,
               data=df.with.binned.vars.added, family='binomial')
summary(model.2)

file.name <- 'model2.html';
stargazer(model.2, type=c('html'),out=paste(out.path,file.name,sep=''),
          title=c('Model #2: Logistic Regression'),
          align=TRUE, digits=2, digits.extra=2, initial.zero=TRUE)

model.score.2 <- model.2$fitted.values
roc.2 <- roc(response=df.with.binned.vars.added$spam, predictor=model.score.2)
print(roc.2)
plot(roc.2)

auc.2 <- auc(roc.2)


coords(roc=roc.2,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.2,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.train$ModelScores <- model.2$fitted.values
spam.train$classes <- ifelse(spam.train$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.train$spam, spam.train$classes)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$classes);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r
#####################################################################
reducedDF <- spam.test %>% 
  select(filterlist)

df.with.binned.vars.added <- woe.binning.deploy(reducedDF, binning,
                                                add.woe.or.dum.var='woe')
df.with.binned.vars.added <- df.with.binned.vars.added %>% 
  select(-filterlist)
names(df.with.binned.vars.added)
df.with.binned.vars.added$spam <- spam.test$spam

model.score.2 <- predict(model.2,df.with.binned.vars.added)
roc.2 <- roc(response=df.with.binned.vars.added$spam, predictor=model.score.2)
print(roc.2)
plot(roc.2)

auc.2 <- auc(roc.2)


coords(roc=roc.2,x=c('best'),
       input=c('threshold','specificity','sensitivity'),
       ret=c('threshold','specificity','sensitivity'),
       as.list=TRUE
)

roc.specs <- coords(roc=roc.2,x=c('best'),
                    input=c('threshold','specificity','sensitivity'),
                    ret=c('threshold','specificity','sensitivity'),
                    as.list=TRUE
)

spam.test$ModelScores <- model.score.2
spam.test$classes <- ifelse(spam.test$ModelScores>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.test$spam, spam.test$classes)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.test$spam, spam.test$classes);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r
###############################################
nb.1 <- naive_bayes(y = as.factor(df.with.binned.vars.added$spam), x= df.with.binned.vars.added %>% select(-'spam'))
names(dataset.woe)
# Compare the table to the table above;
plot(nb.1)

predicted.class <- predict(nb.1,df.with.binned.vars.added)
pct.accuracy <- mean(predicted.class==df.with.binned.vars.added$spam)

predicted.classProb <- predict(nb.1,type=c('prob'))

prob.B <- mean(predicted.class==df.with.binned.vars.added$spam)
prob.NB <- 1-mean(predicted.class==df.with.binned.vars.added$spam)


spam.train$predicted.classNB <- predicted.class

df.1 <- as.data.frame(list(SpamRisk=dataset.woe$spam,Class=dataset.woe %>% select(-'spam')))

spam.train$ModelScoresNB <- predicted.class$fitted.values
spam.train$classesNB <- ifelse(spam.train$ModelScoresNB>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.train$spam, spam.train$predicted.classNB)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.train$spam, spam.train$predicted.classNB);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r
######################test##

reducedDF <- spam.test %>% 
  select(filterlist)


df.with.binned.vars.added <- woe.binning.deploy(reducedDF, binning,
                                                add.woe.or.dum.var='woe')
df.with.binned.vars.added <- df.with.binned.vars.added %>% 
  select(-filterlist)



predicted.class <- predict(nb.1,df.with.binned.vars.added)
pct.accuracy <- mean(predicted.class==dataset.woe$spam)

predicted.classProb <- predict(nb.1,type=c('prob'))

prob.B <- mean(predicted.class==dataset.woe$spam)
prob.NB <- 1-mean(predicted.class==dataset.woe$spam)


spam.test$predicted.classNB <- predicted.class

df.1 <- as.data.frame(list(SpamRisk=dataset.woe$spam,Class=dataset.woe %>% select(-'spam')))

spam.test$ModelScoresNB <- predicted.class$fitted.values
spam.test$classesNB <- ifelse(spam.test$ModelScoresNB>roc.specs$threshold,1,0)

# Rough confusion matrix using counts;
table(spam.test$spam, spam.test$predicted.classNB)
# Note the orientation of the table.  The row labels are the true classes 
# and the column values are the predicted classes;


# Let's create a proper confusion matrix
t <- table(spam.test$spam, spam.test$predicted.classNB);
# Compute row totals;
r <- apply(t,MARGIN=1,FUN=sum);
# Normalize confusion matrix to rates;
t/r

