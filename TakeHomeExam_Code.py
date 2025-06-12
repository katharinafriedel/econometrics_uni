#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:07:58 2024

@author: katharinafriedel
"""

import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import statsmodels.api as sm

data = pd.read_stata('/Users/katharinafriedel/Library/Mobile Documents/com~apple~CloudDocs/EBS/Semester 3 FT24/Econometrics/Take Home Exam/Data-TakeHome-2024Fall/employment_08_09.dta')


# Question A

# determining the different variables
maximum = data.max()
minimum = data.min()
sample_average = data.mean()
std_dev = data.std()
median = data.median()
skewness = data.apply(lambda x: skew(x.dropna()))
kurtosis = data.apply(lambda x: kurtosis(x.dropna()))

# writing the variables into a data frame
descriptive_stats = pd.DataFrame({
    'Sample Average': sample_average,
    'Standard Deviation': std_dev,
    'Median': median,
    'Maximum': maximum,
    'Minimum': minimum,
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

print(descriptive_stats)

# saving the table as an excel file
descriptive_stats.to_excel('descriptive_stats.xlsx', index=True)


# Question B

# determining the data frame for the boxplots
Boxplot = {'DataValue': ['age', 'race', 'earnwke', 'employed', 'unemployed', 'married', 'union', 'ne_states', 'so_states', 'ce_states', 'we_states', 'government', 'private', 'self', 'educ_lths', 'educ_hs', 'educ_somecol', 'educ_aa', 'educ_bac', 'educ_adv', 'female'],
                'Sample Average' :descriptive_stats ['Sample Average'],
                'Standard Deviation' :descriptive_stats ['Standard Deviation'],
                'Median' :descriptive_stats ['Median'],
                'Maximum' :descriptive_stats ['Maximum'],
                'Minimum' :descriptive_stats ['Minimum'],
                'Skewness' :descriptive_stats ['Skewness'],
                'Kurtosis' :descriptive_stats ['Kurtosis']}

df = pd.DataFrame(Boxplot)

# creating the boxplots
plt.figure(figsize = (20,15))           
for index, part in enumerate (df['DataValue']): 
    plt.subplot(5, 5, index + 1)
    plt.boxplot(df.iloc[index, 1:6])
    plt.title(part)

plt.show()
         
        
# Question C

# determining the number of employed workers in April 2009
employed_09 = data[data['employed'] == 1]
print(employed_09)

# dividing the number of employed workers in April 2009 by the number of employed workers in 2008
print(4738/5412,"of workers in the sample were employed in April 2009")

# defining the variables
employed_09 = 4738  
employed_08 = 5412  
employed_09_given_employed_08 = 4738/5412

# determine the standard error and the z-value
se = np.sqrt(employed_09_given_employed_08 * (1 - employed_09_given_employed_08) / employed_08)
z = 1.96  

# calculating the lower and upper value of the confidence interval
ci_lower = employed_09_given_employed_08 - z * se
ci_upper = employed_09_given_employed_08 + z * se

print(f"95% Confidence Interval: ({ci_lower:.4f}, {ci_upper:.4f})")


# Question D

# defining the varaible age_squared
data['age_squared'] = (data['age'].astype('float64')) ** 2

# determining the variables for the regression
Y = data['employed']
X = data[['age', 'age_squared']]
X = sm.add_constant(X)

# creating a Linear Probability model
linear_model = sm.OLS(Y, X).fit()
print(linear_model.summary())

# saving the regression table in csv format (to open in MS Word)
with open('OLSregression.csv', 'w') as f:
    f.write(linear_model.summary().as_csv())

# Question D1

# proving that age is significant that is if its p-value is smaller than 0.05
if linear_model.pvalues['age'] < 0.05:
    print("Age is a statistically significant determinant of employment in April 2009")
else:
    print("Age is not a statistically significant determinant of employment in April 2009")
    
# Question D2

# finding out if there is evidence of a nonlinear effect of age on employment probability, that is if the p-value of age_squared is smaller than 0.05
if linear_model.pvalues['age_squared'] < 0.05:
    print("There is evidence of a nonlinear effect of age on employment probability")
else:
    print("There is no evidence of a nonlinear effect of age on employment probability")

# Question D3

# calculating the predicted employment probability for the different ages
pred_l_20 = linear_model.predict([1, 20, 20**2])[0]
print("The predicted employment probability for a 20-year-old is", pred_l_20)

pred_l_40 = linear_model.predict([1, 40, 40**2])[0]
print("The predicted employment probability for a 40-year-old is", pred_l_40)

pred_l_60 = linear_model.predict([1, 60, 60**2])[0]
print("The predicted employment probability for a 60-year-old is", pred_l_60)


# Question E

# creating a Probit model 
probit_model = sm.Probit(Y, X).fit()
print(probit_model.summary())

# saving the regression table in csv format (to open in MS Word)
with open('Probitregression.csv', 'w') as f:
    f.write(probit_model.summary().as_csv())

# Question E1

# proving that age is significant that is if its p-value is smaller than 0.05
if probit_model.pvalues['age'] < 0.05:
    print("Age is a statistically significant determinant of employment in April 2009")
else:
    print("Age is not a statistically significant determinant of employment in April 2009")
    
# Question E2

# finding out if there is evidence of a nonlinear effect of age on employment probability, that is if the p-value of age_squared is smaller than 0.05
if probit_model.pvalues['age_squared'] < 0.05:
    print("There is evidence of a nonlinear effect of age on employment probability")
else:
    print("There is no evidence of a nonlinear effect of age on employment probability")
    
# Question E3

# calculating the predicted employment probability for the different ages
pred_p_20 = probit_model.predict([1, 20, 20**2])[0]
print("The predicted employment probability for a 20-year-old is", pred_p_20)

pred_p_40 = probit_model.predict([1, 40, 40**2])[0]
print("The predicted employment probability for a 40-year-old is", pred_p_40)

pred_p_60 = probit_model.predict([1, 60, 60**2])[0]
print("The predicted employment probability for a 60-year-old is", pred_p_60)


# Question F

# defining a range of ages
age_range = np.linspace(18, 65)
age_squared_range = age_range ** 2

# creating a data frame 
pred_X = pd.DataFrame({'const': 1, 'age': age_range, 'age_squared': age_squared_range})

# getting the fitted values for each model
lpm_fitted = linear_model.predict(pred_X)
probit_fitted = probit_model.predict(pred_X)

# plotting the fitted values for each model
plt.plot(age_range, lpm_fitted, label="LPM Fitted Values", color="green", linestyle="--")
plt.plot(age_range, probit_fitted, label="Probit Fitted Values", color="orange")

# formatting the graph
plt.xlabel("Age")
plt.ylabel("Predicted Probability of Employment")
plt.title("Comparison of Fitted Values: LPM vs. Probit")
plt.legend()
plt.ylim(0.5, 1)
plt.show()


# Question G

# Question G1

# fixing the issue that most of the results in the table were marked as (nan)
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# creating a Linear Probability model
X1 = sm.add_constant(data[['educ_lths', 'educ_hs', 'educ_somecol', 'educ_aa', 'educ_bac', 'educ_adv', 'female', 'race', 'married', 'ne_states', 'so_states', 'ce_states', 'we_states', 'earnwke']])
lpm_model = sm.OLS(data['employed'], X1).fit()

# creating a Logit model
X2 = sm.add_constant(data[['educ_lths', 'educ_hs', 'educ_somecol', 'educ_aa', 'educ_bac', 'educ_adv', 'female', 'race', 'married', 'ne_states', 'so_states', 'ce_states', 'we_states', 'earnwke']])
logit_model = sm.Logit(data['employed'], X2).fit()

# creating a Probit model
X3 = sm.add_constant(data[['educ_lths', 'ne_states','educ_hs', 'educ_somecol', 'educ_aa', 'educ_bac', 'educ_adv', 'female', 'race', 'married', 'so_states', 'ce_states', 'we_states', 'earnwke']])
probit_model = sm.Probit(data['employed'], X3).fit()

# calculating only the coefficients and p-values for each model
lpm_coef = lpm_model.params
lpm_pvalue = lpm_model.pvalues

logit_coef = logit_model.params
logit_pvalue = logit_model.pvalues

probit_coef = probit_model.params
probit_pvalue = probit_model.pvalues

# combining the results from before (coefficients and p-values) into one data frame
summary_table = pd.DataFrame({
    'LPM Coef.': lpm_coef,
    'LPM P-Value': lpm_pvalue,
    'Logit Coef.': logit_coef,
    'Logit P-Value': logit_pvalue,
    'Probit Coef.': probit_coef,
    'Probit P-Value': probit_pvalue
})

# formatting the table so that the p-values appear in parenthesis
formatted_table = summary_table.applymap(lambda x: f"{x:.3f}")
formatted_table['LPM'] = formatted_table['LPM Coef.'] + ' (' + formatted_table['LPM P-Value'] + ')'
formatted_table['Logit'] = formatted_table['Logit Coef.'] + ' (' + formatted_table['Logit P-Value'] + ')'
formatted_table['Probit'] = formatted_table['Probit Coef.'] + ' (' + formatted_table['Probit P-Value'] + ')'

# putting the columns of the table into my desired order
final_table = formatted_table[['LPM', 'Logit', 'Probit']]

print(final_table)

# saving the final table as an excel file
final_table.to_excel('final_table.xlsx', index=True)


# determining the variables for the hypothesis test
variables = (data[['educ_lths', 'educ_hs', 'educ_somecol', 'educ_aa', 'educ_bac', 'educ_adv', 'female', 'race', 'married', 'ne_states', 'so_states', 'ce_states', 'we_states', 'earnwke']])

# creating the hypothesis test so that it can apply to multiple models
def hypothesis_test(model, model_name):
    print(f"\nHypothesis Test for {model_name} Model:")
    
    # writing out the hypothesis pair
    # H0: coef = 0
    # H1: coef ≠ 0
    
    # determining the p-values
    for var in variables:
        p_value = model.pvalues[var]
        
        # printing the considered variable
        print(f"Variable: {var}")
        
        # determining the interpretation for the results with a 5% type 1 error
        if p_value < 0.05:
            print("Result: Reject the null hypothesis (significant)")
        else:
            print("Result: Fail to reject the null hypothesis (not significant)")
        # printing lines after every test for readability
        print("-" * 40)

# performing hypothesis tests for each model
hypothesis_test(lpm_model, "LPM")
hypothesis_test(logit_model, "Logit")
hypothesis_test(probit_model, "Probit")

# Question G2

# writing the results of the models into one data frame
model_results = {
    'LPM': lpm_model,
    'Logit': logit_model,
    'Probit': probit_model
}

# determining the task
for model_name, result in model_results.items():
    print(f"\nCharacteristics of workers hurt most by the Great Recession in {model_name} model:")
    # variable is relevant if the corresponding pvalue < alpha (0.05)
    for variable in result.params.index:
        if result.pvalues[variable] < 0.05:
            # if the corresponding coefficient is also < 0 than workers with that characteristic (variable) are more hurt by the Great Recession 
            direction = 'more' if result.params[variable] < 0 else 'less'
            print(f"Workers with higher {variable} were {direction} hurt by the Great Recession.")


