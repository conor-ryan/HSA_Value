clear
cd "I:\HSA Probit"
set seed 1234 

// Create study id - state mapping 
import delimited "Data\choice14.csv", clear
collapse (first) state, by(studyid) 

save "Temp\State_StudyID", replace


/// Read in 4-choice data /// 
import delimited "Data\choice14.csv", clear 



* Create plan name variable
gen planname = ""
replace planname="EPO" if planid==1
replace planname="HMO" if planid==2
replace planname="HMO2" if planid==3
replace planname="HRAG" if planid==4
replace planname="HRAS" if planid==5
replace planname="HSA" if planid==6
replace planname="POS" if planid==7
replace planname="PPO" if planid==8

* New Plan ID Fixed Effect
gen plan2 = 0
gen plan3 = 0 
gen plan4 = 0
replace plan2=1 if newpid==2
replace plan3=1 if newpid==3
replace plan4=1 if newpid==4

* New Plan ID Cost Interaction
gen plan2_cost = 0
gen plan3_cost = 0 
gen plan4_cost = 0
replace plan2_cost=cost if newpid==2
replace plan3_cost=cost if newpid==3
replace plan4_cost=cost if newpid==4

* New Plan ID Cost Interaction
gen hra_cost = hra*cost
gen hra_chronic = hra*chronic
gen hra_income = hra*paycheck
gen hra_depend = hra*(dependen>=2)

gen hsa_cost = hsa*cost
gen hsa_chronic = hsa*chronic
gen hsa_income = hsa*paycheck
gen hsa_depend = hsa*(dependen>=2)

gen hmo_cost = hmo*cost
gen hmo_chronic = hmo*chronic
gen hmo_income = hmo*paycheck
gen hmo_depend = hmo*(dependen>=2)



* Age - Premium Interactions
gen price_age_40_60 = 0
gen price_age_60plus = 0
replace price_age_40_60 = adjprem if age>=40 & age<60
replace price_age_60plus = adjprem if age>=60

* Gender and Family Interactions
gen price_female = adjprem*female
gen price_family = adjprem*family

* Log Premium 
gen logprem = log(paycheck - adjprem) //Premiums are annual here, right?//
gen logprice_age_40_60 = 0
gen logprice_age_60plus = 0
replace logprice_age_40_60 = logprem if age>=40 & age<60
replace logprice_age_60plus = logprem if age>=60

* Gender and Family Interactions
gen logprice_female = logprem*female
gen logprice_family = logprem*family

// Drop observations with no income
drop if paycheck<1



save "Temp\choice14_temp", replace
/// Sub-sample for estimation /// 
foreach n in 5 10 20 {
	use "Temp\choice14_temp", clear
	keep if yvar==1 /* One obs per person*/

	sample `n', by(state)

	keep studyid state

	merge 1:m studyid state using "Temp\choice14_temp"
	sort studyid newpid
	keep if _merge==3

	save "Temp\choice14_samp`n'", replace
	outsheet using "Temp\choice14_samp`n'.csv", comma replace
}


/// Read in full choice data /// 
import delimited "Data\choice11.csv", clear 

drop hmo* epo* hmo2* hrag* hras* hsa* pos* ppo* ndhp* cdhp*

merge m:1 studyid using "Temp/State_StudyID" 
drop _merge

* Create plan name variable
gen planname = ""
replace planname="EPO" if planid==1
replace planname="HMO" if planid==2
replace planname="HMO2" if planid==3
replace planname="HRAG" if planid==4
replace planname="HRAS" if planid==5
replace planname="HSA" if planid==6
replace planname="POS" if planid==7
replace planname="PPO" if planid==8

gen hra = planid>=4 & planid<=5
gen hsa = planid==6
gen hmo = planid<=3

* New Plan ID Fixed Effect
gen plan2 = 0
gen plan3 = 0 
gen plan4 = 0
gen plan5 = 0
gen plan6 = 0 
gen plan7 = 0
gen plan8 = 0
replace plan2=1 if planid==2
replace plan3=1 if planid==3
replace plan4=1 if planid==4
replace plan5=1 if planid==5
replace plan6=1 if planid==6
replace plan7=1 if planid==7
replace plan8=1 if planid==8

* New Plan ID Cost Interaction
gen hra_cost = hra*cost
gen hra_chronic = hra*chronic
gen hra_income = hra*paycheck
gen hra_depend = hra*(dependen>=2)

gen hsa_cost = hsa*cost
gen hsa_chronic = hsa*chronic
gen hsa_income = hsa*paycheck
gen hsa_depend = hsa*(dependen>=2)

gen hmo_cost = hmo*cost
gen hmo_chronic = hmo*chronic
gen hmo_income = hmo*paycheck
gen hmo_depend = hmo*(dependen>=2)



* Age - Premium Interactions
gen price_age_40_60 = 0
gen price_age_60plus = 0
replace price_age_40_60 = adjprem if age>=40 & age<60
replace price_age_60plus = adjprem if age>=60

* Gender and Family Interactions
gen price_female = adjprem*female
gen price_family = adjprem*family

* Log Premium 
gen logprem = log(paycheck - adjprem) //Premiums are annual here, right?//
gen logprice_age_40_60 = 0
gen logprice_age_60plus = 0
replace logprice_age_40_60 = logprem if age>=40 & age<60
replace logprice_age_60plus = logprem if age>=60

* Gender and Family Interactions
gen logprice_female = logprem*female
gen logprice_family = logprem*family

// Drop observations with no income
drop if paycheck<1


save "Temp\choice11_temp", replace
/// Sub-sample for estimation /// 
foreach n in 5 10 20 {
	use "Temp\choice11_temp", clear
	keep if yvar==1 /* One obs per person*/

	sample `n', by(state)

	keep studyid state

	merge 1:m studyid state using "Temp\choice11_temp"
	sort studyid planid
	keep if _merge==3

	save "Temp\choice11_samp`n'", replace
	outsheet using "Temp\choice11_samp`n'.csv", comma replace
}
