require(xgboost)
require(methods)


begTime=Sys.time()

cat('Reading train data file ...');flush.console()
train = read.csv('Data/train.csv',header=TRUE,stringsAsFactors = F)
cat(paste('time used:',format(Sys.time()-begTime),'\n'));flush.console()
cat('Reading test data file ...');flush.console()
test = read.csv('Data/test.csv',header=TRUE,stringsAsFactors = F)
cat(paste('time used:',format(Sys.time()-begTime),'\n'));flush.console()
train = train[,-1]
test = test[,-1]

y = train[,ncol(train)]
y = gsub('Class_','',y)
y = as.integer(y)-1 #xgboost take features in [0,numOfClass)

x = rbind(train[,-ncol(train)],test)
x = as.matrix(x)
x = matrix(as.numeric(x),nrow(x),ncol(x))
trind = 1:length(y)
teind = (nrow(train)+1):nrow(x)

myeta=0.2
mydepth=8
mylambda=1
myalpha=0.8
mylambda_bias=0
child_range=c(0.5, 1, 2, 3, 4, 5, 10)
mychild=3
mysubsamp=0.9
mycolsamp=0.6
round_count=3
nround = 150
seednum=20150320
set.seed(seednum)
# Set necessary parameter
param <- list("objective" = "multi:softprob",
        "eval_metric" = "mlogloss",
        "num_class" = 9,
        "bst:eta" = myeta,
        "bst:max_depth" = mydepth,
        "lambda" = mylambda,
        "lambda_bias" = mylambda_bias,
        "alpha" = myalpha,
        "min_child_weight" = mychild,
        "subsample" = mysubsamp,
        "colsample_bytree" = mycolsamp,
        "nthread" = 8)

cat('Train model ...');flush.console()
# Train the model
bst = xgboost(param=param, data = x[trind,], label = y, nrounds=nround)
cat(paste('time used:',format(Sys.time()-begTime),'\n'));flush.console()

cat('Predicting ...');flush.console()
# Make prediction
pred = predict(bst,x[teind,])
pred = matrix(pred,9,length(pred)/9)
pred = t(pred)
cat(paste('time used:',format(Sys.time()-begTime),'\n'));flush.console()

cat('Outputing ...');flush.console()
# Output submission
pred = format(pred, digits=2,scientific=F) # shrink the size of submission
pred = data.frame(1:nrow(pred),pred)
names(pred) = c('id', paste0('Class_',1:9))
fname=paste0('xgb','_nround',nround,'_eta',myeta,'_depth',mydepth,'_lambda',mylambda,
            '_alpha',myalpha,'_child',mychild,'_subsamp',mysubsamp,'_colsamp',mycolsamp,
            '_sub.csv.gz')
write.csv(pred,gzfile(fname), quote=FALSE,row.names=FALSE)
cat(paste('time used:',format(Sys.time()-begTime),'\n'));flush.console()
