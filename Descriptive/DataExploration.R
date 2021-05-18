rm(list=ls())
library(data.table)
library(ggplot2)
library(haven)
setwd("G:/My Drive/HSA Probit")

#### Read in Data ####
df = as.data.table(read.csv("Data/choice11.csv"))
df2 = as.data.table(read.csv("Data/choice14.csv"))
df3 = read_sas("Data/choice11.sas7bdat")

mkt_location = unique(df2[,c("STATE","studyid")])
df = merge(df,mkt_location,all.x=TRUE,by.y="studyid",by.x="STUDYID")

df[PLANID==1,planName:="EPO"]
df[PLANID==2,planName:="HMO"]
df[PLANID==3,planName:="HMO2"]
df[PLANID==4,planName:="HRAG"]
df[PLANID==5,planName:="HRAS"]
df[PLANID==6,planName:="HSA"]
df[PLANID==7,planName:="POS"]
df[PLANID==8,planName:="PPO"]


#### Market Shares ##### 
total = df[,list(enroll=sum(YVAR)),
          by=c("planName")]
total[,share:=enroll/sum(enroll)]


mkts = df[,list(enroll=sum(YVAR),
                premium = mean(EE)),
           by=c("planName","STATE")]
mkts[,totalEnroll:=sum(enroll),by="STATE"]
mkts[,share:=enroll/totalEnroll]


#### HSA SELECTION ####
df[,all_other_premiums:=EE]
df[planName=="HSA",all_other_premiums:=NA]
df[,mean_other_premiums:=mean(all_other_premiums,na.rm=TRUE),by="STUDYID"]

hsa = df[planName=="HSA"]


summary(hsa[,lm(YVAR~EE+mean_other_premiums+COST+CLB+CUB+AGE+PAYCHECK)])

