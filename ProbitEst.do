clear
cd "I:\HSA Probit"
set seed 1234 

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
replace plan2=3 if newpid==3
replace plan2=4 if newpid==4

* New Plan ID Cost Interaction
gen plan2_cost = 0
gen plan3_cost = 0 
gen plan4_cost = 0
replace plan2_cost=cost if newpid==2
replace plan3_cost=cost if newpid==3
replace plan4_cost=cost if newpid==4


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


/// Run some estimation tests ///
use "Temp\choice14_samp5", clear

clogit yvar adjprem hmo_* hmo2_* hsa_* pos_*, group(studyid)

mprobit newpid adjprem cost


asmprobit yvar adjprem, case(studyid) alternatives(newpid) casevars(cost) correlation(unstructured) stddev(homo)
