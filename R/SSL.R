#' Label Propagation
#'
#' @description \code{sslLabelProp} propagates a few known labels to a large number of unknown labels
#' according to their proximities to neighboring nodes. It supports many
#' kinds of distance measurements and graph representations.
#' @param x a n * p matrix or data.frame of n observations and p predictors
#' @param y a vector of k known labels. The rows of \code{y} must be
#' the same as the length of \code{known.label}.
#' @param known.label a vector indicating the row index of known labels in matrix \code{x}.
#' @param graph.type character string; which type of graph should be created? Options
#' include \code{knn},\code{enn},\code{tanh} and \code{exp}.
#' \itemize{\item \code{knn} :kNN graphs.Nodes i, j are connected by an edge if i is in j 's k-nearest-neighborhood. \code{k} is a hyperparameter that controls the density of the graph.
#' \item \code{enn} :epsilon-NN graphs. Nodes i, j are connected by an edge, if the distance \code{d(i, j ) < epsilon.} The  hyperparameter \code{epsilon} controls neighborhood radius.
#' \item \code{tanh}:tanh-weighted graphs.  \code{w(i,j) = (tanh(alpha1(d(i,j) - alpha2)) + 1)/2}. where \code{d(i,j)} denotes the distance between point i and j. Hyperparameters \code{alpha1} and \code{alpha2} control the slope and cutoff value respectively.
#' \item \code{exp} :exp-weighted graphs.\code{w(i,j) = exp(-d(i,j)^2/alpha^2)},where \code{d(i,j)} denotes the distance between point i and j. Hyperparameter \code{alpha} controls the decay rate.}
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param alpha numeric parameter needed when \code{graph.type = exp}
#' @param alpha1 numeric parameter needed when \code{graph.type = tanh}
#' @param alpha2 numeric parameter needed when \code{graph.type = tanh}
#' @param k integer parameter needed when \code{graph.type = knn }
#' @param epsilon numeric parameter needed when \code{graph.type = enn}
#' @param iter iteration
#' @return a n * 1 vector indicating the predictions of n observations in C class
#' @author Junxiang Wang
#' @seealso \code{\link{dist}}, \code{\link{pr_DB}}
#' @details
#' \code{sslLabelProp} implements label propagation algorithm in \code{iter} iterations.It
#' supports many kinds of distance measurements and four types of graph creations.
#' @export
#' @importFrom proxy dist
#' @importFrom NetPreProc Prob.norm
#' @examples
#' data(iris)
#' x<-iris[,1:4]
#' #Suppose we know the first twenty observations of each class and we want to propagate
#' #these labels to unlabeled data.
#' # 1 setosa, 2 versicolor, 3 virginica
#' y<-rep(1:3,each =20)
#' known.label <-c(1:20,51:70,101:120)
#' f1<-sslLabelProp(x,y,known.label,graph.type="enn",epsilon = 0.5)
#' f2<-sslLabelProp(x,y,known.label,graph.type="knn",k =10)
#' f3<-sslLabelProp(x,y,known.label,graph.type="tanh",alpha1=-2,alpha2=1)
#' f4<-sslLabelProp(x,y,known.label,graph.type="exp",alpha = 1)
#' @references Xiaojin Zhu(2005),Semi-Supervised Learning with Graphs
sslLabelProp <-
  function(x,y,known.label,graph.type = "exp",dist.type = "Euclidean",alpha,alpha1,alpha2,
           k,epsilon,iter = 1000)
  {
    all.Obs = dim(x)[1]
    all.label = unique(y)
    C <- length(unique(y))
    num.known = length(known.label)
    if (num.known != length(y))
      stop("the number of known.label  doesn't accord with that of y")
    num.class = dim(y)[2]
    d <- proxy::dist(x,x,method = dist.type)
    d <- matrix(d,ncol = all.Obs)
    if (graph.type == "knn")
    {
      index <- sapply(1:all.Obs,function(i)
      {
        return(sort(d[,i],index.return = TRUE)$ix[1:k])
      })
      w <- matrix(0,ncol = all.Obs,nrow = all.Obs)
      for (i in 1:all.Obs)
      {
        w[index[,i],i] <- d[index[,i],i]
      }
    }
    if (graph.type == "enn")
      w <- ifelse(d < epsilon,d,0)
    if (graph.type == "tanh")
      w <- (tanh(alpha1 * (d - alpha2)) + 1) / 2
    if (graph.type == "exp")
      w <- exp(-d ^ 2 / alpha ^ 2)
    p <- NetPreProc::Prob.norm(w)
    p <- t(p)
    rm(w,d)
    ff <- rep(0,all.Obs)
    ff[known.label] <- y
    for (i in 1:iter)
    {
      ff <- ff %*% p
      ff[known.label] <- y
    }
    return (as.vector(ff))
  }
#' Co-Training
#'
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n * 1 integer vector of labels.
#' @param xu a m * p matrix or data.frame of unlabeled data
#' @param method1,method2 a string which specifies the first and second classification model to use.\code{xgb} means extreme gradient boosting,please refer to \code{\link{xgb.train}}.For other options,see more in \code{\link{train}}.
#' @param nrounds1,nrounds2 parameter needed when \code{method1} or \code{method2} =\code{xgb}. See more in \code{\link{xgb.train}}
#' @param portion the percentage of data to split into two parts.
#' @param n the number of unlabeled examples to add into label data in each iteration.
#' @param seed an integer specifying random number generation state for data split
#' @param ... other parameters
#' @return a m * 1 integer vector representing the predictions  of  unlabeled data.
#' @author Junxiang Wang
#' @details \code{sslCoTrain} divides labeled data into two parts ,each part is trained with a classifier,
#' then it chooses some unlabeled examples for prediction and adds them into labeled data. These new labeled data
#' help the other classifer improve performance.
#' @seealso \code{\link{train}} \code{\link{xgb.train}}
#' @export
#' @importFrom caret createDataPartition
#' @importFrom caret train
#' @import xgboost
#' @import klaR
#' @import e1071
#' @examples
#' data(iris)
#' xl<-iris[,1:4]
#' #Suppose we know the first twenty observations of each class
#' #and we want to predict the remaining with co-training
#' # 1 setosa, 2 versicolor, 3 virginica
#' yl<-rep(1:3,each=20)
#' known.label <-c(1:20,51:70,101:120)
#' xu<-xl[-known.label,]
#' xl<-xl[known.label,]
#' yu<-sslCoTrain(xl,yl,xu,method1="xgb",nrounds1 = 100,method2="xgb",nrounds2 = 100,n=60)
#' @references Blum, A., & Mitchell, T. (1998). Combining labeled and unlabeled data with co-training. COLT: Proceedings of the Workshop on Computational Learning Theory.
sslCoTrain <-
  function(xl,yl,xu,method1 = "nb",method2 = "nb",nrounds1,nrounds2,portion =
             0.5,n = 10,seed = 0,...)
  {
    yu <- NULL
    num.class <-length(unique(yl)) + 1
    while (dim(xu)[1] != 0)
    {
      yl <- as.factor(yl)
      set.seed(seed)
      seq <- createDataPartition(y = yl,list = FALSE,p = portion)
      x1 <- xl[seq,]
      y1 <- yl[seq]
      x2 <- xl[-seq,]
      y2 <- yl[-seq]
      num <- min(dim(xu)[1],floor(n / 2))
      xd <- xu[1:num,]
      xu <- xu[-(1:num),]
      if (method1 == "xgb")
      {

        dtrain <- xgb.DMatrix(data = as.matrix(x1),label = y1)
        h1 <- xgb.train(data = dtrain,nrounds = nrounds1,num_class =num.class,objective ='multi:softmax',...)
        pred <- predict(h1,as.matrix(xd))
        pred <- round(pred)
      }
      else{
        h1 <- train(x1,y1,method = method1,...)
        pred <- predict(h1,xd)
      }
      yu <- c(yu,pred)
      xl <- rbind(xl,xd)
      yl <- c(yl,pred)
      num <- min(dim(xu)[1],n - floor(n / 2))
      if (num > 0)
      {
        xd <- xu[1:num,]
        xu <- xu[-(1:num),]
        if (method2 == "xgb")
        {
          dtrain <- xgb.DMatrix(data = as.matrix(x2),label = y2)
          h2 <- xgb.train(data = dtrain,nrounds = nrounds2,num_class =num.class,objective ='multi:softmax',...)
          pred <- predict(h2,as.matrix(xd))
          pred <- round(pred)
        }
        else{
          h2 <- train(x2,y2,method = method2,...)
          pred <- predict(h2,xd)
        }
        yu <- c(yu,pred)
        xl <- rbind(xl,xd)
        yl <- c(yl,pred)
      }
    }
    return(yu)
  }
#'Self-Training
#'
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n * 1 integer vector of labels(begin from 1).
#' @param xu a m * p matrix or data.frame of unlabeled data
#' @param n number of unlabeled examples to add into labeled data in each iteration
#' @param nrounds the maximal number of iterations, see more in \code{\link{xgb.train}}
#' @param ... other parameters
#' @return a m * 1 integer vector representing the predictions  of  unlabeled data.
#' @author Junxiang Wang
#' @details  In self-training a classifier is first trained with the small amount of labeled data using
#' extreme gradient boosting. The classifier is then used to classify the unlabeled data. The most confident
#'unlabeled points, together with their predicted labels, are added to the training
#'set. The classifier is re-trained and the procedure repeats.
#'@examples
#' data(iris)
#' xl<-iris[,1:4]
#' #Suppose we know the first twenty observations of each class
#' #and we want to predict the remaining with self-training
#' # 1 setosa, 2 versicolor, 3 virginica
#' yl<-rep(1:3,each = 20)
#' known.label <-c(1:20,51:70,101:120)
#' xu<-xl[-known.label,]
#' xl<-xl[known.label,]
#' yu<-sslSelfTrain(xl,yl,xu,nrounds = 100,n=30)
#' @export
#' @import xgboost
#' @references Rosenberg, C., Hebert, M., & Schneiderman, H. (2005). Semi-supervised selftraining of object detection models. Seventh IEEE Workshop on Applications of
#' Computer Vision.
#' @seealso \code{\link{xgb.train}}
sslSelfTrain <- function(xl,yl,xu,n = 10,nrounds,...)
{
  yu <- NULL
  seq <- NULL
  num.class<-length(unique(yl))+1
  all.obs <-dim(xu)[1]
  remain <-1:all.obs
  while ((is.null(seq))||(dim(xu[-seq,])[1] != 0))
  {
    yl <- as.factor(yl)
    num <- min(dim(xu)[1],n)
    dtrain <- xgb.DMatrix(data = as.matrix(xl),label = yl)
    h <- xgb.train(data = dtrain,nrounds = nrounds,num_class =num.class,objective ='multi:softprob',...)
    if(is.null(seq))
    {
      pred <- matrix(predict(h,as.matrix(xu)),ncol=num.class,byrow =T)
    }
    else
    {
    pred <- matrix(predict(h,as.matrix(xu[-seq,])),ncol=num.class,byrow =T)
    }
    pred <-pred[,-1]
    label<-sapply(1:dim(pred)[1],function(x){
      which.max(pred[x,])
    })
    label.prob <- sapply(1:dim(pred)[1],function(x){
      max(pred[x,])
    })
    new <- sort(label.prob,decreasing = T,index.return = T)$ix[1:num]
    xl <- rbind(xl,xu[remain[new],])
    yl <- c(yl,label[new])
    yu <- c(yu,label[new])
    seq<-c(seq,remain[new])
    remain <-setdiff(remain,seq)
  }
  seq<-sort(seq,index.return = T)$ix
  yu<-yu[seq]
  return(yu)
}

#' Gaussian Mixture Model with an EM Algorithm
#' @description \code{sslGmmEM} implements Gaussian Mixture Model with an EM algorithm,
#' and weights the unlabeled data by introducing lambda-EM technique.
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n * 1 integer vector of labels.
#' @param xu a m * p matrix or data.frame of unlabeled data
#' @param seed an integer specifying random number generation state for spliting labeled data into training set and cross-validation set.
#' @param improvement numeric. Minimal allowed improvement of parameters.
#' @param p percentage of labeled data are splitted into cross-validation set.
#' @return a list of values is returned:
#' @field  para a numeric estimated parameter matrix in which the column represents variables and the row represents estimated means and standard deviation of each class.
#' for example, the first and second row represents the mean and standard deviation of the first class, the third and fourth row represents  the mean and standard deviation of the second class,etc.
#' @field classProb the estimated class probabilities
#' @field yu the predicted label of unlabeled data
#' @field optLambda the optimal lambda chosen by cross-validation
#' @details
#' \code{sslGmmEM} introduces unlabeled data into parameter estimation process. The weight \code{lambda} is chosen by cross-validation.
#' The Gaussian Mixture Model is estimated based on maximum log likelihood function with an EM algorithm. The E-step
#' computes the probabilities of each class for every observation. The M-step computes parameters based on probabilities
#' obtained in the E-step.
#' @author Junxiang Wang
#' @export
#' @importFrom caret createDataPartition
#' @importFrom stats dnorm var
#' @examples
#' data(iris)
#' xl<-iris[,-5]
#' #Suppose we know the first twenty observations of each class
#' #and we want to predict the remaining with Gaussian Mixture Model
#' #1 setosa, 2 versicolor, 3 virginica
#'yl<-rep(1:3,each=20)
#'known.label <-c(1:20,51:70,101:120)
#'xu<-xl[-known.label,]
#'xl<-xl[known.label,]
#'l<-sslGmmEM(xl,yl,xu)
#' @references Kamal Nigam, Andrew Mccallum, Sebastian Thrun, Tom Mitchell(1999) Text Classification from Labeled and Unlabeled Documents using EM
#'
sslGmmEM <- function(xl,yl,xu,seed = 0,improvement = 10e-5,p = 0.3)
{
  all.label <- unique(yl)
  label.var <- dim(xl)[2]
  optAcc <- 0
  optLambda <- 0
  # data split
  set.seed(seed)
  seq <- createDataPartition(y = yl,p = p,list = F)
  xtrain <- xl[-seq,]
  ytrain <- yl[-seq]
  xcv <- xl[seq,]
  ycv <- yl[seq]
  #initalization
  classProbInit <- sapply(all.label,function(x)
  {
    return(sum(ytrain == x) / length(ytrain))
  })
  paraInit <- sapply(1:label.var,function(x)
  {
    col <- xl[,x]
    resu <- NULL
    for (i in all.label)
    {
      index <- which(yl == i)
      temCol <- col[index]
      resu <- c(resu,mean(temCol),sqrt(var(temCol)))
    }
    return(resu)
  })
  #choose lambda with cross-validation
  for (lambda in seq(0,1,0.01))
  {
    para <- paraInit
    classProb <- classProbInit
    # train paramters with an EM algorithm
    emList <- EM(para,xtrain,xu,ytrain,all.label,classProb,lambda)
    para <- emList$para
    classProb <- emList$classProb
    # test performance
    ycvVal <- nbPred(para,xcv,all.label,classProb)
    index <- sapply(1:length(ycv),function(x)
    {
      return(which.max(ycvVal[x,]))
    })
    label <- all.label[index]
    acc <- sum(label == ycv) / length(ycv)
    if (optAcc <= acc)
    {
      optAcc <- acc
      optLambda <- lambda
    }
  }
  # paramter estimation with optimal lambda
  paraOld <- paraInit
  emList <- EM(paraInit,xl,xu,yl,all.label,classProbInit,optLambda)
  para <- emList$para
  classProb <- emList$classProb
  while (norm(para - paraOld,type = "F") >= improvement)
  {
    paraOld <- para
    emList <- EM(para,xl,xu,yl,all.label,classProb,optLambda)
    para <- emList$para
    classProb <- emList$classProb
  }
  yuVal <- nbPred(para,xu,all.label,classProb)
  index <- sapply(1:dim(yuVal)[1],function(x)
  {
    return(which.max(yuVal[x,]))
  })
  yu <- all.label[index]
  return(list(
    para = para,classProb = classProb,yu = yu,optLambda = optLambda
  ))
}

nbTrain <- function(ylVal,yuVal,xl,xu,lambda,all.label)
{
  all.var <- dim(xl)[2]
  para <- sapply(1:all.var,function(x)
  {
    col <- c(xl[,x],xu[,x])
    resu <- NULL
    for (i in all.label)
    {
      tempVal <- c(ylVal[,i],lambda * yuVal[,i])
      m <- sum(tempVal * col) / sum(tempVal)
      sd <- sqrt(sum(tempVal * (col - m) ^ 2) / sum(tempVal))
      resu <- c(resu,m,sd)
    }
    return(resu)
  })
  return(para)
}

nbPred <- function(para,xcv,allLabel,classProb)
{
  val <- NULL
  for (i in 1:length(allLabel))
  {
    prob <- sapply(1:dim(xcv)[2],function(x)
    {
      col <- xcv[,x]
      return(dnorm((col - para[2 * i - 1,x]) / para[2 * i,x]))
    })
    val <- cbind(val,exp(rowSums(log(prob))) * classProb[i])
  }
  return(val)
}
EM <- function(para,xl,xu,yl,all.label,classProb,lambda)
{
  yuVal <- nbPred(para,xu,all.label,classProb)
  colYu <- rowSums(yuVal)
  yuVal <- yuVal / colYu
  ylVal <- sapply(all.label, function(x) {
    col <- numeric(length = length(yl))
    index <- which(yl == x)
    col[index] <- 1
    return(col)
  })
  index <- sapply(1:dim(yuVal)[1],function(x)
  {
    return(which.max(yuVal[x,]))
  })
  yu <- all.label[index]
  para <- nbTrain(ylVal,yuVal,xl,xu,lambda,all.label)
  classProb <- sapply(all.label,function(x)
  {
    return((sum(yl == x) + lambda * sum(yu == x)) / (length(yl) + lambda *
                                                       length(yu)))
  })
  return(list(para = para,classProb = classProb))
}

#' Low Density Separation
#' @description \code{sslLDS} implements low density separation with Transductive Support Vector Machines(TSVM) for semi-supervised binary classification
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n * 1 binary labels(1 or -1).
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param rho numeric;a parameter for connectivity kernel.It defines  minimal rho-path distances.
#' @param C numeric; a parameter in the TSVM training model.
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param p the percentage of data used for cross-validation set.
#' @param improvement numeric; minimal allowed improvement of parameters.
#' @param seed an integer specifying random number generation state for spliting labeled data into training set and cross-validation set.
#' @param delta numeric; a allowed cutoff for the cumulative percent of variance to lose by multidimensional scaling.
#' @param alpha numeric; a learning rate in the gradient descent algorithm.
#' @return a list of values is returned:
#' @field yu the predicted label of unlabeled data
#' @field optC.star the optimal C.star chosen by cross-validation. C.star weights the unlabeled data in the TSVM model.
#' @field para estimated parameters of TSVM, including \code{w} and \code{b}
#' @details
#' \code{sslLDS} constructs a low density graph with connectivity kernel.It implements multidemensional scaling
#' for demensionality reduction and chooses optimal \code{C.star} by cross-validation. Finally, it trains the TSVM model with gradient descent algorithm.
#' @author Junxiang Wang
#' @export
#' @importFrom caret createDataPartition
#' @importFrom  proxy dist
#' @importFrom  Rcpp evalCpp
#' @useDynLib SSL
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70),-5]
#' xu<-iris[c(21:50,71:100),-5]
#' yl<-rep(c(1,-1),each=20)
#' l<-sslLDS(xl,yl,xu,alpha=0.1)
#' @references Chapelle, O., & Zien, A. (2005) Semi-supervised classification by low density separation.In Proceedings of the tenth international workshop on artificial intelligence and statistics.(pp. 57-64). Barbados.

sslLDS<-function(xl,yl,xu,rho = 1,C = 1,dist.type ="Euclidean",p =0.3,improvement = 10e-5,seed=0,delta=0.01,alpha =0.01)
{
 x<-rbind(xl,xu)
 all.obs <-dim(x)[1]
 #construct a low density graph
 d <- proxy::dist(x,x,method = dist.type)
 w<-exp(d)-1
dsp<-Floyd(w,all.obs)
D <-(log(1+dsp)/rho)^2
#multidemensional scaling
H <-D-1/all.obs
M<--H%*%D%*%t(H)
E<-eigen(M,symmetric = T)
U<-E$vectors
L<-E$values
rm(d,w,dsp,D,H,E)
s <-sum(ifelse(L>0,L,0))

logic <-sapply(1:all.obs,function(x)
  { if((sum(L[1:x])>(1-delta)*s)&(L[x]<=delta*L[1]))
     return(T)
else return(F)})
minp<-which(logic)[1]
newX<-sapply(1:minp,function(i) {
  col<-U[,i]/sqrt(L[i])
  if(max(col)!=min(col))
  col<-(col-min(col))/(max(col)-min(col))
  else col<-rep(0,length(col))
  return(col)
    })
#data split
set.seed(seed)
seq <-createDataPartition(y=yl,p=p,list=F)
ytrain <-yl[-seq]
xtrain <-newX[-seq,]
ycv <-yl[seq]
xcv <-newX[seq,]
optAcc<-0
optC.star<-0
#choose C.star by cross-validation
for(i in 1:10)
{
  C.star <- 2^(i-10)*C
  w <-TSVM(xtrain,ytrain,C,C.star,minp,improvement,alpha)
  b <-mean(ytrain)-mean(xtrain%*%w)
  pred <-ifelse(xcv%*%w + b>0,1,-1)
  acc<-sum(pred==ycv)/length(ycv)
  if(optAcc <= acc)
  {
    optAcc <-acc
    optC.star <- C.star
  }
}
#training
w <-TSVM(newX,yl,C,optC.star,minp,improvement,alpha)
b <-mean(yl)-mean(newX%*%w)
yu <-ifelse(newX[-(1:length(yl)),]%*%w + b>0,1,-1)
return(list(yu =yu,optC.star=optC.star,para =list(w=w,b=b)))
}


TSVM<-function(xtrain,ytrain,C,C.star,minp,improvement,alpha)
{
w<-rep(0,minp)
wDiff<-dif(xtrain,ytrain,w,C,C.star,minp)
iter <-0
n<-length(ytrain)
while(sqrt(sum(wDiff^2))>=improvement)
{
  iter<-iter + 1
  w <- w - alpha*sqrt(1/iter)*wDiff
  wDiff<-dif(xtrain,ytrain,w,C,C.star,minp)
  }
return(w)
}

dif<-function(x,yl,w,C,C.star,minp)
{
  n<-length(yl)
  temp <-x%*%w
  m<-mean(yl)
  tu<-mean(temp[-(1:n)])
  diff<- w + 2*C*colSums((yl*(temp[1:n]+m-tu)-1)*yl*(x[1:n,]-colMeans(x[-(1:n),])))
  + C.star *colSums(-6*exp(-3*(temp[-(1:n)]+m-tu)^2)*(temp[-(1:n)]+m-tu)*x[-(1:n),])
  return(diff)
}

Floyd <- function(cost, n) {
  .Call('SSL_Floyd', PACKAGE = 'SSL', cost, n)
}
#' Mincut
#' @description \code{sslMincut} implements the Mincut algorithm for maxflow graph partition in the k-nearest neighbor graph.
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n * 1 binary labels(1 or -1).
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param simil.type character string; this parameter controls the type of similarity measurement.(see \code{\link{simil}} or \code{\link{pr_DB}}).
#' @param k an integer parameter controls a k-nearest neighbor graph.
#' @return  a m * 1 integer vector representing the predicted labels  of  unlabeled data.
#' @author Junxiang Wang
#' @export
#' @importFrom proxy simil
#' @importFrom  Rcpp evalCpp
#' @useDynLib SSL
#' @references Blum, A., & Chawla, S. (2001). Learning from labeled and unlabeled data using
#' graph mincuts. Proc. 18th International Conf. on Machine Learning.
#' @details \code{sslMincut} creates a k-nearest neighbor graph and finds a maxflow
#' from the first postive observation to the first negative one based on MPLA algorithm. This
#' maxflow partitions the graph into postive labels and negative ones.
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70),-5]
#' xu<-iris[c(21:50,71:100),-5]
#' yl<-rep(c(1,-1),each=20)
#' yu<-sslMincut(xl,yl,xu)
#' @seealso \code{\link{pr_DB}} \code{\link{simil}}
sslMincut<-function(xl,yl,xu,simil.type="correlation",k = 10)
{
  num.label <-length(yl)
  index.pos<-which(yl==1)
  index.neg <-which(yl==-1)
  first.pos<-min(index.pos)
  first.neg<-min(index.neg)
  x<-rbind(xl,xu)
  all.Obs <-dim(x)[1]
  s<-proxy::simil(x,x,method = simil.type)
  mi<-min(s)
  ma<-max(s)
  diag(s)<-mi
  s<-(s-mi)/(ma-mi)
  s[first.pos,index.pos]<-0
  s[index.pos,first.pos]<-0
  s[first.neg,index.neg]<-0
  s[index.neg,first.neg]<-0
  index <- sapply(1:all.Obs,function(i)
  {
    return(sort(s[,i],index.return = TRUE,decreasing = T)$ix[1:k])
  })
  sim <- matrix(0,ncol = all.Obs,nrow = all.Obs)
  for (i in 1:all.Obs)
  {
    sim[index[,i],i] <- s[index[,i],i]
  }
  rm(s,index,all.Obs,index.pos,index.neg,x,mi,ma)
  l<-MPLA(sim,first.pos-1,first.neg-1)
  yu<-l[-(1:num.label)]
  return(yu)
}
MPLA <- function(r, s, t) {
  .Call('SSL_MPLA', PACKAGE = 'SSL', r, s, t)
}
#' Local	and	Global	Consistency
#' @param xl a n * p matrix or data.frame of labeled data
#' @param yl a n X C matrix representing  labels of n observations in C classes.If observation i belongs to class j, then yl(i,j)=1, and other elements in the same row equal 0.
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param alpha a numeric parameter controls convergence rate.
#' @param gamma a numeric parameter  in the affinity matrix
#' @param iter the number of iteration.
#' @return  a m * 1 integer vector representing the predicted labels  of  unlabeled data.
#' @references Zhou, D., Bousquet, O., Lal, T., Weston, J. and Scholkopf, B. (2004). Learning with local and global consistency.
#' @export
#' @importFrom proxy dist
#' @author Junxiang Wang
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70,101:120),-5]
#' yl<-matrix(0,ncol=3,nrow=60)
#' yl[1:20,1]<-1
#' yl[21:40,2]<-1
#' yl[41:60,3]<-1
#' xu<-iris[-c(1:20,51:70,101:120),-5]
#' yu<-sslLLGC(xl,yl,xu)
#' @seealso \code{\link{pr_DB}} \code{\link{dist}}
sslLLGC<-function(xl,yl,xu,dist.type="Euclidean",alpha=0.01,gamma=1,iter = 1000)
{
  x<-rbind(xl,xu);
  f<-matrix(0,nrow=dim(x)[1],ncol = dim(yl)[2])
  f[1:dim(yl)[1],]<-yl
  y<-f
  d<-proxy::dist(x,x,method=dist.type)
  W<-exp(-d^2/(2*gamma^2))
  rm(d)
  # compute D^-1/2
  D<-diag(1/sqrt(rowSums(W)))
  S<-D%*%W%*%D
  for(i in 1:iter)
  {
    f <-alpha*S%*%f +(1-alpha)*y
  }
yu<-sapply((dim(xl)[1]+1):dim(x)[1],function(x)
  {return(which.max(f[x,]))})
return(yu)
}
#' t-step Markov Random Walks
#' @param xl a n * p matrix or data.frame of labeled data.
#' @param yl a n * 1 binary labels(1 or -1).
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param t step size.
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param k an integer parameter controls a k-nearest neighbor graph.
#' @param gamma a numeric parameter  in the affinity matrix.
#' @param improvement numeric. Maximum allowed distance between computed parameters in two successive iterations at the steady state.
#' @return  a m * 1 integer vector representing the predicted labels  of  unlabeled data.
#' @details \code{sslMarkovRandomWalks} transmits known labels to unlabeled data by t-step Markov random walks.Parameters are estimated by  an EM algorithm.
#' @author Junxiang Wang
#' @export
#' @importFrom proxy dist
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70),-5]
#' xu<-iris[c(21:50,71:100),-5]
#' yl<-rep(c(1,-1),each=20)
#' yu<-sslMarkovRandomWalks(xl,yl,xu)
#' @seealso \code{\link{pr_DB}} \code{\link{dist}}
#' @references Szummer, M., & Jaakkola, T. (2001). Partially labeled classification with M
#'random walks. Advances in Neural Information Processing Systems, 14.
sslMarkovRandomWalks<-function(xl,yl,xu,t=10,dist.type="Euclidean",k=10,gamma = 1,improvement=10e-5)
{
x<-rbind(xl,xu)
all.Obs<-dim(x)[1]
known.label<-length(yl)
d<-proxy::dist(x,x,method=dist.type)
index <- sapply(1:all.Obs,function(i)
{
  return(sort(d[,i],index.return = TRUE)$ix[1:k])
})
w <- matrix(0,ncol = all.Obs,nrow = all.Obs)
for (i in 1:all.Obs)
{
  w[index[,i],i] <- exp(-d[index[,i],i]/gamma^2)
}
diag(w)<-1
At<-w/rowSums(w)
prob<-At
for(i in 2:t)
At<-At%*%prob
#  probabilities for  positive labels
P<-numeric(all.Obs)
P.old<-P
pos.index<-which(yl==1)
P[pos.index]<-1
neg.index<-setdiff(1:known.label,pos.index)
P[-(1:known.label)]<-0.5
cSums<-colSums(At)
#EM
while(sum((P-P.old)^2)>improvement)
{
  P.old<-P
  P.pos<-At*P
  P<-colSums(P.pos)/cSums
  P[pos.index]<-1
  P[neg.index]<-0
}
P<-ifelse(P>=0.5,1,-1)
yu<-as.vector(P)
yu<-yu[-(1:known.label)]
return(yu)
}

#' Regression on graphs
#' @description \code{sslRegress} develops a regularization framework on graphs.It supports many
#' kinds of distance measurements and graph representations. However, it only supports binary classifications.
#' @param xl a n * p matrix or data.frame of labeled data.
#' @param yl a n * 1 binary labels(1 or -1).
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param graph.type character string; which type of graph should be created? Options
#' include\code{tanh} and \code{exp}.
#' \itemize{\item \code{tanh}:tanh-weighted graphs.  \code{w(i,j) = (tanh(alpha1(d(i,j) - alpha2)) + 1)/2}.where \code{d(i,j)} denotes the distance between point i and j. Hyperparameters \code{alpha1} and \code{alpha2} control the slope and cutoff value respectively.
#' \item \code{exp} :exp-weighted graphs.\code{w(i,j) = exp(-d(i,j)^2/alpha^2)},where \code{d(i,j)} denotes the distance between point i and j. Hyperparameter \code{alpha} controls the decay rate.}
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param alpha numeric parameter needed when \code{graph.type = exp}
#' @param alpha1 numeric parameter needed when \code{graph.type = tanh}
#' @param alpha2 numeric parameter needed when \code{graph.type = tanh}
#' @param p an ineger parameter controls the power of Laplacian for regularization.
#' @param method character string; this parameter choose two possible algorithms:"Tikhonov" means  Tikhonov regularization;"Interpolated" means Interpolated regularization.
#' @param gamma a  parameter of Tikhonov regularization.
#' @return  a m * 1 integer vector representing the predicted labels  of  unlabeled data(1 or -1).
#' @author Junxiang Wang
#' @export
#' @importFrom proxy dist
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70),-5]
#' xu<-iris[c(21:50,71:100),-5]
#' yl<-rep(c(1,-1),each=20)
#' # Tikhonov regularization
#' yu1<-sslRegress(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1)
#' yu2<-sslRegress(xl,yl,xu,graph.type="exp",alpha = 1)
#' # Interpolated regularization
#' yu3<-sslRegress(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1,method="Interpolated")
#' yu4<-sslRegress(xl,yl,xu,graph.type="exp",alpha = 1,method="Interpolated")
#' @seealso \code{\link{pr_DB}} \code{\link{dist}}
#' @references Belkin, M., Matveeva, I., & Niyogi, P. (2004a). Regularization and semisupervised learning on large graphs. COLT
sslRegress<-function(xl,yl,xu,graph.type="exp",dist.type="Euclidean",alpha,alpha1,alpha2,
                    p=2,method="Tikhonov",gamma=1)
{
  x<-rbind(xl,xu)
  all.Obs<-nrow(x)
  d <- proxy::dist(x,x,method = dist.type)
  d <- matrix(d,ncol = all.Obs)
  if (graph.type == "tanh")
    w <- (tanh(alpha1 * (d - alpha2)) + 1) / 2
  if (graph.type == "exp")
    w <- exp(-d ^ 2 / alpha ^ 2)
  d<-diag(colSums(w))
  L<-d-w
  rm(x,d,w)
  k<-length(yl)
  meanY<-mean(yl)
  y1<-c(yl,rep(0,all.Obs-k))
  s<-rep(1,all.Obs)
  S<-diag(s)
  for(i in 1:p)
    S<-S%*%L
  rm(L)
  if(method =="Tikhonov")
  {
    temp<-solve(k*gamma*S+diag(c(rep(1,k),rep(0,all.Obs-k))))
        mu<-(-1)*(s%*%(temp%*%y1))/(s%*%(temp%*%s))
    f<-temp%*%(y1+mu)
    yu<-ifelse(f[-(1:k)]>0,1,-1)
  }
  if(method =="Interpolated")
  {
    S2<-S[1:k,-(1:k)]
    S3<-S[-(1:k),-(1:k)]
    t<-matrix(s[-(1:k)],nrow=1)
    temp<--solve(S3)%*%t(S2)
    y2<-as.matrix(yl-meanY)
    mu<-as.numeric((-1)*(t%*%(temp%*%y2))/(t%*%(temp%*%s[1:k])))
    f<-temp%*%(y2+mu)
    f<-as.vector(f)
    yu<-ifelse(f>0,1,-1)
  }
return(yu)
}
#' Laplacian Regularized Least Squares
#' @param xl a n * p matrix or data.frame of labeled data.
#' @param yl a n * 1 binary labels(1 or -1).
#' @param xu a m * p matrix or data.frame of unlabeled data.
#' @param graph.type character string; which type of graph should be created? Options
#' include \code{knn},\code{enn},\code{tanh} and \code{exp}.
#' \itemize{\item \code{knn} :kNN graphs.Nodes i, j are connected by an edge if i is in j 's k-nearest-neighborhood. \code{k} is a hyperparameter that controls the density of the graph.
#' \item \code{enn} :epsilon-NN graphs. Nodes i, j are connected by an edge, if the distance \code{d(i, j ) < epsilon}. The  hyperparameter \code{epsilon} controls neighborhood radius.
#' \item \code{tanh}:tanh-weighted graphs.  \code{w(i,j) = (tanh(alpha1(d(i,j) - alpha2)) + 1)/2}.where \code{d(i,j)} denotes the distance between point i and j. Hyperparameters \code{alpha1} and \code{alpha2} control the slope and cutoff value respectively.
#' \item \code{exp} :exp-weighted graphs.\code{w(i,j) = exp(-d(i,j)^2/alpha^2)},where \code{d(i,j)} denotes the distance between point i and j. Hyperparameter \code{alpha} controls the decay rate.}
#' @param dist.type character string; this parameter controls the type of distance measurement.(see \code{\link{dist}} or \code{\link{pr_DB}}).
#' @param alpha numeric parameter needed when \code{graph.type = exp}
#' @param alpha1 numeric parameter needed when \code{graph.type = tanh}
#' @param alpha2 numeric parameter needed when \code{graph.type = tanh}
#' @param k integer parameter needed when \code{graph.type = knn }
#' @param epsilon numeric parameter needed when \code{graph.type = enn}
#' @param kernel character string; it controls four types of common kernel functions:\code{linear},\code{polynomial},\code{gaussian} and \code{sigmoid}.
#' \itemize{\item \code{linear}:Linear kernel;\code{k(x,y)=dot(x,y)+c1},where \code{dot(x,y)} is the dot product of vector x and y,\code{c1} is a constant term.
#' \item \code{polynomial}:Polynomial kernel;\code{k(x,y)=(alpha3 *dot(x,y)+c2)^deg},where \code{dot(x,y)} is the dot product of vector x and y.Adjustable parameters are the slope \code{alpha3}, the constant term \code{c2} and the polynomial degree \code{deg}.
#' \item \code{gaussian}:Gaussian kernel;\code{k(x,y)=exp(-gamma*d(x,y)^2)},where \code{d(x,y)} is Euclidean distace between vector x and y,\code{gamma} is a slope parameter.
#' \item \code{sigmoid}:Hyperbolic Tangent (Sigmoid) Kernel;\code{k(x,y)=tanh(alpha4*dot(x,y)+c3)},where \code{d(x,y)} is dot product of vector x and y.There are two adjustable parameters in the sigmoid kernel, the slope \code{alpha4} and the intercept constant \code{c3}.}
#' @param c1 numeric parameter needed when \code{kernel = linear}
#' @param c2 numeric parameter needed when \code{kernel = polynomial}
#' @param c3 numeric parameter needed when \code{kernel = sigmoid}
#' @param deg integer parameter needed when \code{kernel = polynomial}
#' @param gamma numeric parameter needed when \code{kernel = gaussian}
#' @param alpha3 numeric parameter needed when \code{kernel = polynomial}
#' @param alpha4 numeric parameter needed when \code{kernel = sigmoid}
#' @param gammaA numeric; model parameter.
#' @param gammaI numeric; model parameter.
#' @return  a m * 1 integer vector representing the predicted labels  of  unlabeled data(1 or -1).
#' @author Junxiang Wang
#' @export
#' @importFrom proxy dist
#' @examples
#' data(iris)
#' xl<-iris[c(1:20,51:70),-5]
#' xu<-iris[c(21:50,71:100),-5]
#' yl<-rep(c(1,-1),each=20)
#' # combinations of different graph types and kernel types
#' # graph.type =knn, kernel =linear
#' yu1<-sslLapRLS(xl,yl,xu,graph.type="knn",k=10,kernel="linear",c1=1)
#' # graph.type =knn, kernel =polynomial
#' yu2<-sslLapRLS(xl,yl,xu,graph.type="knn",k=10,kernel="polynomial",c2=1,deg=2,alpha3=1)
#' # graph.type =knn, kernel =gaussian
#' yu3<-sslLapRLS(xl,yl,xu,graph.type="knn",k=10,kernel="gaussian",gamma=1)
#' # graph.type =knn, kernel =sigmoid
#' yu4<-sslLapRLS(xl,yl,xu,graph.type="knn",k=10,kernel="sigmoid",c3=-10,
#' alpha4=0.001,gammaI  = 0.05,gammaA = 0.05)
#' # graph.type =enn, kernel =linear
#' yu5<-sslLapRLS(xl,yl,xu,graph.type="enn",epsilon=1,kernel="linear",c1=1)
#' # graph.type =enn, kernel =polynomial
#' yu6<-sslLapRLS(xl,yl,xu,graph.type="enn",epsilon=1,kernel="polynomial",c2=1,deg=2,alpha3=1)
#' # graph.type =enn, kernel =gaussian
#' yu7<-sslLapRLS(xl,yl,xu,graph.type="enn",epsilon=1,kernel="gaussian",gamma=1)
#' # graph.type =enn, kernel =sigmoid
#' yu8<-sslLapRLS(xl,yl,xu,graph.type="enn",epsilon=1,kernel="sigmoid",c3=-10,
#' alpha4=0.001,gammaI  = 0.05,gammaA = 0.05)
#' # graph.type =tanh, kernel =linear
#' yu9<-sslLapRLS(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1,kernel="linear",c1=1)
#' # graph.type =tanh, kernel =polynomial
#' yu10<-sslLapRLS(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1,
#' kernel="polynomial",c2=1,deg=2,alpha3=1)
#' # graph.type =tanh, kernel =gaussian
#' yu11<-sslLapRLS(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1,kernel="gaussian",gamma=1)
#' # graph.type =tanh, kernel =sigmoid
#' yu12<-sslLapRLS(xl,yl,xu,graph.type="tanh",alpha1=-2,alpha2=1,
#' kernel="sigmoid",c3=-10,alpha4=0.001,gammaI  = 0.05,gammaA = 0.05)
#' # graph.type =exp, kernel =linear
#' yu13<-sslLapRLS(xl,yl,xu,graph.type="exp",alpha=1,kernel="linear",c1=1)
#' # graph.type =exp, kernel =polynomial
#' yu14<-sslLapRLS(xl,yl,xu,graph.type="exp",alpha=1,kernel="polynomial",c2=1,deg=2,alpha3=1)
#' # graph.type =exp, kernel =gaussian
#' yu15<-sslLapRLS(xl,yl,xu,graph.type="exp",alpha=1,kernel="gaussian",gamma=1)
#' # graph.type =exp, kernel =sigmoid
#' yu16<-sslLapRLS(xl,yl,xu,graph.type="exp",alpha=1,kernel="sigmoid",
#' c3=-10,alpha4=0.001,gammaI  = 0.05,gammaA = 0.05)
#' @seealso \code{\link{pr_DB}} \code{\link{dist}}
#' @references Olivier Chapelle, Bernhard Scholkopf and Alexander Zien (2006). Semi-Supervised Learning.The MIT Press.
sslLapRLS<-function(xl,yl,xu,graph.type="exp",dist.type="Euclidean",alpha,alpha1,alpha2,
                    k,epsilon,kernel="gaussian",c1,c2,c3,deg,gamma,alpha3,alpha4,gammaA=1,gammaI=1)
{
  x<-rbind(xl,xu)
  all.Obs<-nrow(x)
  l<-length(yl)
  d <- proxy::dist(x,x,method = dist.type)
  d <- matrix(d,ncol = all.Obs)
  # graph type
  if (graph.type == "knn")
  {
    index <- sapply(1:all.Obs,function(i)
    {
      return(sort(d[,i],index.return = TRUE)$ix[1:k])
    })
    w <- matrix(0,ncol = all.Obs,nrow = all.Obs)
    for (i in 1:all.Obs)
    {
      w[index[,i],i] <- d[index[,i],i]
    }
  }
  if (graph.type == "enn")
    w <- ifelse(d < epsilon,d,0)
  if (graph.type == "tanh")
    w <- (tanh(alpha1 * (d - alpha2)) + 1) / 2
  if (graph.type == "exp")
    w <- exp(-d ^ 2 / alpha ^ 2)
  d<-diag(colSums(w))
  L<-d-w
  rm(d,w)
  J<-diag(c(rep(1,l),rep(0,all.Obs-l)))
  I<-diag(rep(1,all.Obs))
  #kernel function
  if(kernel=="linear")
  {
    x<-t(x)
    K<-crossprod(x,x)+c1
  }
  if(kernel=="polynomial")
  {
    x<-t(x)
    temp<-alpha3*crossprod(x,x)+c2
    K<-diag(rep(1,ncol(temp)))
    for(i in 1:deg)
      K<-K%*%temp
    rm(temp)
  }
  if(kernel=="gaussian")
  {
    d <- proxy::dist(x,x,method = "Euclidean")
    d <- matrix(d,ncol = all.Obs)
    K<-exp(-gamma*d)
    rm(d)
  }
  if(kernel=="sigmoid")
  {
    x<-t(x)
    K<-tanh(alpha4*crossprod(x,x)+c3)
  }
  rm(x)
  Y<-c(yl,rep(0,all.Obs-l))
  alphaStar<-solve(J%*%K+gammaA*l*I +gammaI*l/(all.Obs^2)*L%*%K)%*%Y
  f<-K[-(1:l),]%*%alphaStar
  f<-as.vector(f)
  yu<-ifelse(f>0,1,-1)
  return(yu)
}
