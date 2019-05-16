library(caret)

#Data Preparation
data=iris[which(iris$Species!='setosa'),]
data$Species=as.numeric(data$Species)
data$Species=data$Species-2
data=as.matrix(data)

y_index=ncol(data)

#Placeholder for test & train accuracy
trainData_prediction=rep(1,100)
tstData_prediction=rep(1,100)

# Execulute 100 times and later average the accuracy
for(count in c(1:100))
{

  #Split the data in train & test set
  set.seed(count)
  split=createDataPartition(y=data[,y_index], p=0.7, list=FALSE)

  training_data=data[split,]
  test_data=data[-split,]

  training_x=training_data[,-y_index]
  training_y=training_data[,y_index]

  #Normalize Train Data
  tr_ori_mean <- apply(training_x,2, mean)
  tr_ori_sd   <- apply(training_x,2, sd)

  tr_offsets <- t(t(training_x) - tr_ori_mean)
  tr_scaled_data  <- t(t(tr_offsets) / tr_ori_sd)

  #Get Positive class Index
  positive_idx = which(training_data[,y_index] == 1)


  positive_data = tr_scaled_data[positive_idx,]
  negative_data = tr_scaled_data[-positive_idx,]


  #Get Means and SD on Scaled Data
  pos_means=apply(positive_data,2,mean)
  pos_sd=apply(positive_data,2,sd)

  neg_means=apply(negative_data,2,mean)
  neg_sd=apply(negative_data,2,sd)

  test_x=test_data[,1:y_index-1]

  predict_func=function(test_x_row){

    target=0;

    #Used dnorm() function for normal distribution and calculate probability
    p_pos=sum(log(dnorm(test_x_row,pos_means,pos_sd)))+log(length(positive_idx)/length(training_y))
    p_neg=sum(log(dnorm(test_x_row,neg_means,neg_sd)))+log( 1 - (length(positive_idx)/length(training_y)))

    if(p_pos>p_neg){
      target=1
    }else{
      target=0
    }
  }

  #Scale Test Data
  tst_offsets <- t(t(test_x) - tr_ori_mean)
  tst_scaled_data  <- t(t(tst_offsets) / tr_ori_sd)

  #Predict for test data, get prediction for each row
  y_pred=apply(tst_scaled_data,1,predict_func)
  target=test_data[,y_index]

  tstData_prediction[count]=length(which((y_pred==target)==TRUE))/length(target)

  #Predict for train data ( optional, output not printed )
  y_pred_train=apply(tr_scaled_data,1,predict_func)

  trainData_prediction[count]=length(which((y_pred_train==training_y)==TRUE))/length(training_y)

}
print(paste("Average Train Data Accuracy:",mean(trainData_prediction)*100.0,sep = " "))
print(paste("Average Test Data Accuracy:",mean(tstData_prediction)*100.0,sep = " "))




