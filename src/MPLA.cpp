#include <Rcpp.h>
#include <queue>
#include<vector>
#include<cmath>
#define epsilon 10e-5
using namespace Rcpp;
using namespace std;

IntegerVector dfs(int s,int t,IntegerVector level, NumericMatrix r,int size)
{ queue<int> q;
  IntegerVector label=IntegerVector(size);
  q.push(s);
  label(s) =1;
  level(s) =0;
  while(!q.empty())
  {
    int v=q.front();
     q.pop();
    for(int i=0;i<size;i++)
    {
      if((fabs(r(v,i))>epsilon)&&(!label(i)))
       { q.push(i);
         label(i)=1;
         level(i)=level(v) + 1;
         if(i==t)
         {break;}
      }
    }
    if(label(t))
      break;
  }
  return(level);
}
vector<int> findPath(NumericMatrix r,int s,int t,IntegerVector level,int size)
{
  vector<int> p;
  p.push_back(t);
  int temp =t;
  for(int i=level(t)-1;i!=level(s);i--)
  {
      for(int j=0;j<size;j++)
      {
        if((level(j)==i)&&(fabs(r(j,temp))>epsilon))
        {
          temp =j;
          break;
        }
      }
    p.push_back(temp);
  }
  p.push_back(s);
  return(p);
}

// [[Rcpp::export]]
IntegerVector MPLA(NumericMatrix r,int s,int t) {
int size =r.nrow();
// initialization
IntegerVector level = IntegerVector(size,-1);
IntegerVector label = IntegerVector(size);
level =dfs(s,t,level,r,size);

while(level[t]!=-1)
{
    vector<int> p=findPath(r,s,t,level,size);
    if(p.size()>0)
    {//compute the bottleneck capacity for the flow
      float C =9999;
      for(vector<int>::iterator it=p.end()-1;it!=p.begin();it--)
      {
        if(C>r(*it,*(it-1)))
         C =r(*it,*(it-1));
      }
       //update residual network
      for(vector<int>::iterator it=p.end()-1;it!=p.begin();it--)
      {
        r(*it,*(it-1)) -=C;
        r(*(it-1),*it) +=C;
      }

  }
level = IntegerVector(size,-1);
level =dfs(s,t,level,r,size);
}
// label nodes
for(int i=0;i<level.size();i++)
{
if(level(i)==-1)
{label(i)=-1;}
else
  {label(i)=1;}
}
return(label);
}
