using CSV
df = CSV.read("$googleDrivePath/HSA Probit/Temp/choice14_samp5.csv")

df[!,:plan2].=0.0
df[!,:plan2][df[!,:newpid].==2] .= 1.0

df[!,:plan3].=0.0
df[!,:plan3][df[!,:newpid].==3] .= 1.0

df[!,:plan4].=0.0
df[!,:plan4][df[!,:newpid].==4] .= 1.0
