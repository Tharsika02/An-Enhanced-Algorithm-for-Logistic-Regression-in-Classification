library(MASS)
library(caret)



setwd('C:\\Users\\ADMIN\\OneDrive\\Desktop\\Research\\Research\\Code\\Datasets')
# Read the data from CSV
df1 <- read.csv("creditcard.csv")
df <- na.omit(df1)

y_counts <- table(df$class)
count_y11 <- y_counts[1]  # Count of y=0
count_y01 <- y_counts[2]  # Count of y=1
cat("Count of y=0:", count_y11, "\n")
cat("Count of y=1:", count_y01, "\n")


set.seed(123) 
train_indices <- sample(nrow(df), 0.7 * nrow(df))
train_data <- df[train_indices, ]
test_data <- df[-train_indices, ]

X <- train_data[, -29]
X1 <- as.big.matrix(X1)
# Create the response variable y
y <- train_data[[29]]


y_counts <- table(train_data$class)
count_y1 <- y_counts[1]  # Count of y=0
count_y0 <- y_counts[2]  # Count of y=1
cat("Count of y=0:", count_y1, "\n")
cat("Count of y=1:", count_y0, "\n")

# Create the design matrix X
X1 <- test_data[, -29]
X1 <- as.big.matrix(X1)
y1 <- test_data[[29]]


LASSO=function(X,y,epsilon=1e-8){
  
  p=dim(X)[2]
  #initial step

  beta.hat=rep(0,p)
  pi0 <- exp(X %*% beta.hat) / (1 + exp(X %*% beta.hat))
  W <- diag(c(pi0 * (1 - pi0)))
  C1 <- t(X) %*% W %*% X
  eta0 <- X %*% beta.hat
  n <- rep(1,dim(X)[1])
  z1 <- eta0 + (y - pi0) / (pi0 * (n - pi0))
  r=z1
  alpha=0
  i=1
  l=1
  
  
  #compute inner product (correlations between y & X) and choose the first variable to enter the model
  C=t(X)%*%r
  s=sign(C)
  indX=which.max(abs(C)) # index of the variable to enter the model
  I=diag(p)
  E=I[,indX]
  
  
  while(alpha<1){
    
    beta.hat=rbind(beta.hat,rep(0,p))
    
    
    d=ginv((t(E)%*%t(X))%*%W%*%(X%*%E))%*%(t(E)%*%t(X))%*%W%*%r #equiangular vector 
    Xd=X%*%E%*%d
    alpha.pm=matrix(rep(1,3*p),nrow=3,ncol=p)
    for (j in 1:p) {
      
      if(j%in%indX==FALSE){
        if((C[indX[length(indX)]]-C[j])/(C[indX[length(indX)]]-t(X[,j])%*%Xd)>epsilon){
          alpha.pm[1,j]=(C[indX[length(indX)]]-C[j])/(C[indX[length(indX)]]-t(X[,j])%*%Xd)}
        if((C[indX[length(indX)]]+C[j])/(C[indX[length(indX)]]+t(X[,j])%*%Xd)>epsilon){
          alpha.pm[2,j]=(C[indX[length(indX)]]+C[j])/(C[indX[length(indX)]]+t(X[,j])%*%Xd)}
      }
      
      if((i>1 & j%in%indX)==TRUE){
        if(-beta.hat[i,j]/d[i]>epsilon){
          alpha.pm[3,j]=-beta.hat[i,j]/d[i]}
      }
      
    }
    alpha=min(alpha.pm)
    
    if(which(alpha.pm == alpha, arr.ind = TRUE)[1]<3){
      indX[i+1]=which(alpha.pm == alpha, arr.ind = TRUE)[2]
      
      for(j in 1:i){
        beta.hat[l+1,indX[j]]=beta.hat[l,indX[j]]+alpha*d[j]
      }
      E=cbind(E,I[,which(alpha.pm == alpha, arr.ind = TRUE)[2]])
      i=i+1
    }
    else{
      #print("Enter into LASSO")
      E=E[,-which(indX == which(alpha.pm == alpha, arr.ind = TRUE)[2], arr.ind = TRUE)]
      indX=indX[-which(indX == which(alpha.pm == alpha, arr.ind = TRUE)[2], arr.ind = TRUE)]
      beta.hat[l+1,which(alpha.pm == alpha, arr.ind = TRUE)[2]]=0
      i=i-1
      for(j in 1:i){
        beta.hat[l+1,indX[j]]=beta.hat[l,indX[j]]+alpha*d[j]
      }
      
    }
    
    r=r-alpha*Xd
    C=t(X)%*%r
    l=l+1
    
    
  }
  #print("LASSO Solutions\n")
  #print(W)
  #list(beta=beta.hat[l,])
  return(beta.hat[l,])
  
}

result_final=LASSO(X,y,epsilon=1e-8)
result_final

# Classify instances based on predicted probabilities
predicted_prob <- X1 %*% result_final
predicted_class <- ifelse(predicted_prob >= 0.5, 1, 0)
actual_class <- y1


# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(predicted_class == 1 & actual_class == 1)
FP <- sum(predicted_class == 1 & actual_class == 0)
FN <- sum(predicted_class == 0 & actual_class == 1)
TN <- sum(predicted_class == 0 & actual_class == 0)

# Calculate Precision, Recall, and F1 score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy <- (sensitivity + specificity) / 2

# Print the results for IRLS _ LASSO
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("specificity:", specificity, "\n")
cat("Accuracy:",accuracy,"\n") 
cat("F1 Score:", f1_score, "\n")
cat("Balanced Accuracy:", balanced_accuracy, "\n")

#Decision Tree :

library(rpart)

#  Build the decision tree model using the rpart function
model2_DTree <- rpart(class ~ .- 1, data = train_data, method = "class")

#  Make predictions on the test data using the decision tree model
predicted_class <- predict(model2_DTree, test_data, type = "class")
actual_class <- test_data$class

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(predicted_class == 1 & actual_class == 1)
FP <- sum(predicted_class == 1 & actual_class == 0)
FN <- sum(predicted_class == 0 & actual_class == 1)
TN <- sum(predicted_class == 0 & actual_class == 0)

# Calculate Precision, Recall, and F1 score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy <- (sensitivity + specificity) / 2

# Print the results for DECISION TREE
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("specificity:", specificity, "\n")
cat("Accuracy:",accuracy,"\n") 
cat("F1 Score:", f1_score, "\n")
cat("Balanced Accuracy:", balanced_accuracy, "\n")



# Support vector machine :

# Load the required libraries
library(e1071)
library(caret)

# Convert target variable to factor
train_data$class <- as.factor(train_data$class)
test_data$class <- as.factor(test_data$class)

#  Build the SVM model using e1071
model3_SVM <- svm(class ~ ., data = train_data)

# Predict using the testing set

predicted_class <- predict(model3_SVM, test_data)
actual_class <- test_data$class

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(predicted_class == 1 & actual_class == 1)
FP <- sum(predicted_class == 1 & actual_class == 0)
FN <- sum(predicted_class == 0 & actual_class == 1)
TN <- sum(predicted_class == 0 & actual_class == 0)

# Calculate Precision, Recall, and F1 score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy <- (sensitivity + specificity) / 2

# Print the results for Support vector machine
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("specificity:", specificity, "\n")
cat("Accuracy:",accuracy,"\n") 
cat("F1 Score:", f1_score, "\n")
cat("Balanced Accuracy:", balanced_accuracy, "\n")



# Naive bayse :


# Load the required libraries
library(naivebayes)
library(caret)


# Convert target variable to factor
train_data$class <- as.factor(train_data$class)
test_data$class <- as.factor(test_data$class)

#  Build the Naive Bayes model using naivebayes
model4_naive_bayes <- naive_bayes(class ~ ., data = train_data)

#Predict using the testing set
predicted_class <- predict(model4_naive_bayes, newdata = test_data)
actual_class <- test_data$class

# Calculate True Positives (TP), False Positives (FP), and False Negatives (FN)
TP <- sum(predicted_class == 1 & actual_class == 1)
FP <- sum(predicted_class == 1 & actual_class == 0)
FN <- sum(predicted_class == 0 & actual_class == 1)
TN <- sum(predicted_class == 0 & actual_class == 0)

# Calculate Precision, Recall, and F1 score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
sensitivity <- TP / (TP + FN)
specificity <- TN / (TN + FP)
accuracy <- (TP + TN) / (TP + TN + FP + FN)
balanced_accuracy <- (sensitivity + specificity) / 2

# Print the results for Naive bayse
cat("Precision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("specificity:", specificity, "\n")
cat("Accuracy:",accuracy,"\n") 
cat("F1 Score:", f1_score, "\n")
cat("Balanced Accuracy:", balanced_accuracy, "\n")


