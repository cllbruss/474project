## Set up
library(regclass)
library(gbm)
library(caret)
library(e1071)
library(pROC)
library(rpart)
library(randomForest)

## Load Data in
DATA <- read.csv("data.csv")
head(DATA)

## Replace Empty Values
DATA$RM[is.na(DATA$RM)] <- mean(DATA$RM, na.rm=TRUE)

## Transform Data
DATA$CRIM <- log10(DATA$CRIM + 1.00632)
DATA$ZN <- log10(DATA$ZN + 1)
DATA$INDUS <- log10(DATA$INDUS + 1.46)
DATA$CHAS <- log10(DATA$CHAS + 1)
DATA$NOX <- log10(DATA$NOX + 1.385)
DATA$AGE <- log10(DATA$AGE)
DATA$DIS <- log10(DATA$DIS)
DATA$RAD <- log10(DATA$RAD) 
DATA$TAX <- log10(DATA$TAX)
DATA$PTRATIO <- log10(DATA$PTRATIO)
DATA$B <- log10(DATA$B + 1.32)
DATA$LSTAT <- log10(DATA$LSTAT)
DATA$MEDV <- log10(DATA$MEDV)

## Training and Holdout
set.seed(2021)
train.rows <- sample(1:nrow(DATA), 0.7 * nrow(DATA))
TRAIN <- DATA[train.rows,]
HOLDOUT <- DATA[-train.rows,]

## fitControl
fitControl <- trainControl(method = "cv", number = 5, verboseIter = FALSE)

##rPart MEDV~. AND MEDV~CRIM
set.seed(2021)
paramGrid <- expand.grid(cp = 10^seq(-5, -1.5, length = 25))

TREE <- train(MEDV~., data = TRAIN, method = "rpart", trControl = fitControl, tuneGrid = paramGrid, preProc = c("center","scale"))

TREE$results[rownames(TREE$bestTune),]
TREE$results
varImp(TREE)
plot(TREE)

TREE1 <- rpart(MEDV~., data=TRAIN, cp = 0.004217)  
visualize_model(TREE1)

set.seed(2021)
TREE2 <- train(MEDV~CRIM, data = TRAIN, method = "rpart", trControl = fitControl, tuneGrid = paramGrid, preProc = c("center","scale"))

TREE2$results[rownames(TREE2$bestTune),]
TREE2$results
plot(TREE2)

TREE3 <- rpart(MEDV~CRIM, data=TRAIN, cp = 0.022603)  
visualize_model(TREE3)


## Multiple Regression for the relationship between CRIM and LSTAT
HEX <- lm(CRIM~LSTAT, data = DATA)
summary(HEX)
hist(HEX)
plot(HEX)

## Multiple Regression for the relationship between CRIM and LSTAT
TECH <- lm(INDUS~RAD, data = DATA)
summary(TECH)
plot(TECH)




##OTHER METHODS CONSIDERED


## GLM
GLM <- train(MEDV~., data = TRAIN, method = "glm", trControl=fitControl, preProc= c('center', 'scale'))

GLM$results

##GLMNET
glmnetGrid <- expand.grid(alpha = seq(0,1,.1),lambda = 10^seq(-5,-1,by=0.25))

GLMNET <- train(MEDV~., data = TRAIN, method = "glmnet", tuneGrid=glmnetGrid, trControl = fitControl, preProc= c('center', 'scale'))

GLMNET$results[rownames(GLMNET$bestTune),]
plot(GLMNET)

## Random Forest
forestGrid <- expand.grid(mtry=c(1,3,5))

FOREST <- train(MEDV~., data=TRAIN , method='rf', tuneGrid= forestGrid, trControl= fitControl, preProc=c('center','scale'))
FOREST$results[rownames(FOREST$bestTune),]
plot(FOREST)





