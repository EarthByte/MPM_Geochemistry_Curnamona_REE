#This code is used to implement ILR transformation and RPCA.
#Load packages
library(rrcov)
library(factoextra)
library(corrplot)
# Load the dataset
f=read.csv("D:/Dataset/ILRRPCA.csv")  
#select data
sel=c(3:29) 
x1=f[,sel]
#Define ILR transformation
ilr <- function(x){
  x.ilr=matrix(NA,nrow=nrow(x),ncol=ncol(x)-1)
  for (i in 1:ncol(x.ilr)){
    x.ilr[, i]=sqrt((i)/(i+1))*log(((apply(as.matrix(x[,1:i]), 1,
                                           prod))^(1/i))/(x[,i+1]))
  }
  return(x.ilr)
}
x2=ilr(x1)
#Perform RPCA
x2.mcd=covMcd(x2,cor=TRUE)
summary(x2.mcd)
resrob=princomp(x2,covmat=x2.mcd,cor=TRUE)
#Visualize the result of RPCA
get_eigenvalue(resrob)
fviz_eig(resrob,addlabels = T,ylim=c(0,25))
resrob.var <- get_pca_var(resrob)
resrob.var$cor
resrob.var$coord  
resrob.var$cos2
resrob.var$contrib
fviz_pca_var(resrob,col.var = "contrib",gradient.cols=c("#00AFBB","#E7B800","#FC4E07"),axes=1:2)
corrplot(resrob.var$coord,is.corr=F,tl.cex = 0.75)
corrplot(resrob.var$cos2,is.corr=F,tl.cex = 0.75)
corrplot(resrob.var$contrib,is.corr=F,tl.cex = 0.75)
fviz_contrib(resrob, choice = "var", axes = 1,tl.cex = 0.75)
fviz_contrib(resrob, choice = "var", axes = 1:2,tl.cex = 0.75)
fviz_cos2(resrob, choice = "var",axes = 1,tl.cex = 0.75)
#Save loadings and scores
write.csv(resrob$loadings,"D:/loadings.csv") 
write.csv(resrob$scores,"D:/Scores.csv") 

