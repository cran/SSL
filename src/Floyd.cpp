#include <Rcpp.h>
using namespace Rcpp;
//[[Rcpp::export]]
NumericMatrix Floyd(NumericMatrix  cost,int n)
{
   NumericMatrix dist(cost);
   for(int k=0;k<n;k++)
      for(int i=0;i<n;i++)
        for(int j=i+1;j<n;j++)
          if(dist(i,j)>dist(i,k)+dist(k,j))
          {
            dist(i,j)=dist(i,k)+dist(k,j);
            dist(j,i)=dist(i,j);
          }

return(dist);
}

