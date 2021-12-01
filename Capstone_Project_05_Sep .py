#!/usr/bin/env python
# coding: utf-8

# <table align="center" width=100%>
#     <tr>
#         <td>
#             <div align="center">
#                 <font color="#21618C" size=8px>
#                     <b> Capstone Project <br> Heart Attack
#                     </b>
#                 </font>
#             </div>
#         </td>
#     </tr>
# </table>

# ### About the data set  (Heart Attack data)
# 
# The dataset contains information about several health and economic factors that contribute to Heart attack <br> Attribute information: 
# 
# **Patient_ID:** Unique ID of different patients.
# 
# **Gender:** Gender of the patient.
# 
# **Age:** Age of the patient.
# 
# **HyperTension:** A person got HyperTension or not
# 
# **Heart_Disease:** A person got affected with Heart_Disease
# 
# **Is_Married:** Hepatitis B (HepB) immunization coverage for 1 year olds (Percentage)
# 
# **Employment_Type:** Determines whether the patient is a working professional in a Private/Govt sectors, never worked or children.
# 
# **Residential_type:** Specifies whether the patient is from Urban/Rural areas.
# 
# **Glucose_Levels:** Average glucose levels of a patient.
# 
# **BMI_Values:** Considering height and weight of a patient.
# 
# **Smoking_Habits:** Classifies whether the patient is a regular smoker, past smoker ornever smoked.
# 
# **Heart_Attack:** Chances of getting heart attack (Dependent Variable)
# 
# 

# In[1]:


import pandas as pd 

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
 
import seaborn as sns

from warnings import filterwarnings
filterwarnings('ignore')


from sklearn.model_selection import train_test_split

import statsmodels
import statsmodels.api as sm

from sklearn.preprocessing import StandardScaler 

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

from sklearn.feature_selection import RFE


# In[2]:


df=pd.read_csv('Heart_attack.csv')
df.head()


# In[3]:


df[['Age','Glucose_Levels','BMI_Values']].describe().T


# In[4]:


df.info()


# In[5]:


df.isnull().sum()


# In[6]:


df.apply(lambda x : len(x.unique()))


# ### Gender

# In[7]:


sh1=df["Gender"].value_counts()
print(sh1)
sns.set(style="darkgrid")
sns.barplot(sh1.index,sh1.values,alpha=0.7)


# ### Smoking Habits

# In[8]:


sh=df["Smoking_Habits"].value_counts()
print(sh)
sns.set(style="darkgrid")
sns.barplot(sh.index,sh.values)


# ### Residential_type

# In[9]:


sh2=df["Residential_type"].value_counts()
print(sh2)
sns.set(style="darkgrid")
sns.barplot(sh2.index,sh2.values)


# ### Employment_Type

# In[10]:


sh3=df['Employment_Type'].value_counts()
print(sh3)
sns.set(style='darkgrid')
sns.barplot(sh3.index,sh3.values)


# ### Is_Married

# In[11]:


sh4=df['Is_Married'].value_counts()
print(sh4)
sns.set(style='darkgrid')
sns.barplot(sh4.index,sh4.values)


# In[12]:


sns.heatmap(df.isnull(), cbar=False)


# In[13]:


cat_df = df.select_dtypes(include=['object']).copy()
cat_df.head()


# In[14]:


cat_df.isnull().sum()


# In[15]:


df.columns


# ### HyperTension

# In[16]:


sh5=df['HyperTension'].value_counts()
print(sh5)
sns.barplot(sh5.index,sh5.values)


# ### Heart_Disease

# In[17]:


sh6=df['Heart_Disease'].value_counts()
print(sh6)
sns.barplot(sh5.index,sh5.values)


# ### BMI_Values

# In[18]:


sns.kdeplot(df['BMI_Values'])


# ### Glucose_Levels

# In[19]:


#Glucose_Levels
sns.distplot(df['Glucose_Levels'])


# ### Age

# In[20]:


sns.kdeplot(df['Age'])


# In[21]:


df.columns


# In[22]:


df['BMI_Values'].skew()


# In[23]:


a=df['BMI_Values'].median()
df['BMI_Values']=df['BMI_Values'].fillna(a)


# In[24]:


df.isnull().sum()


# In[25]:


df['Smoking_Habits']=df['Smoking_Habits'].fillna('never smoked')


# In[26]:


df.isnull().sum()


# In[27]:


sns.distplot(df['BMI_Values'])


# In[28]:


df['BMI_Values'].skew()


# In[29]:


np.log(df['BMI_Values']).skew()


# In[30]:


sns.boxplot(np.log(df['BMI_Values']))


# ## Bivariate Analysis

# In[31]:


sns.scatterplot('BMI_Values', 'Glucose_Levels', data = df);


# In[32]:


sns.barplot(x='Smoking_Habits',y='BMI_Values', data=df)


# In[33]:


sns.barplot(x='Smoking_Habits',y='Glucose_Levels', data=df)


# In[34]:


sns.barplot(x='Residential_type',y='Glucose_Levels', data=df)


# In[35]:


sns.barplot(x='Residential_type',y='BMI_Values', data=df)


# In[36]:


sns.barplot(data=df, x='Residential_type', y='BMI_Values', hue='Gender')


# In[37]:


sns.barplot(data=df, x='Residential_type', y='Glucose_Levels', hue='Gender')


# In[38]:


sns.barplot(data=df, x='Is_Married', y='BMI_Values', hue='Gender')


# In[39]:


sns.barplot(data=df, x='Is_Married', y='Glucose_Levels', hue='Gender')


# In[40]:


sns.barplot(data=df, x='Smoking_Habits', y='BMI_Values', hue='Gender')


# In[41]:


sns.barplot(data=df, x='Smoking_Habits', y='Glucose_Levels', hue='Gender')


# ## Multivariate Analysis

# In[42]:


sns.pairplot(df, diag_kind = 'kde')

plt.show()


# ## Distribution of Variables

# In[43]:


df.drop('Heart_Attack', axis = 1).hist()
plt.tight_layout()
plt.show()  
print('Skewness:')
df.drop('Heart_Attack', axis = 1).skew()


# In[ ]:





# ## Removing Outliers

# In[44]:


df1=df.copy()


# In[45]:


df1.head()


# In[46]:


df1[['HyperTension','Heart_Disease']] = df1[['HyperTension','Heart_Disease']].astype('object')#@
df1.dtypes


# In[47]:


df1.shape


# In[48]:


df1['Heart_Attack'].value_counts()


# In[49]:


df1=df1.reset_index(drop=True)


# In[50]:


df1.shape


# In[51]:


df1['Heart_Attack'].value_counts()


# In[52]:


#df1['BMI_Values'].plot(kind='box')
sns.boxplot(df1['BMI_Values'])


# In[53]:


df1['Glucose_Levels'].skew()


# In[54]:


sns.boxplot(df1['Glucose_Levels'])


# In[55]:


df1=df1.reset_index(drop=True)


# In[56]:


#7  indep cate var of object d type must be there.in that hyp and hd ka d type must be changed from int to obj.@df_Cat


# In[57]:


df1_cat = df1[['Gender','HyperTension','Heart_Disease','Is_Married','Employment_Type','Residential_type','Smoking_Habits']]
df1_cat.head()#all are pure cate var with o,1 or str values.


# In[58]:


df1_cat.shape


# In[59]:


#df1_cat..earlier it had 7 cate var.now it has 12 encoded col related to 7 cate var.


# In[60]:


df1_cat= pd.get_dummies(df1_cat,drop_first = True)
df1_cat.head()#df1_cat has dummy var of all cate var(including hyp and hdisease) .df1 and df same both have dep var too


# In[61]:


df1_cat.shape


# In[62]:


num_df1 = df1.drop(['Patient_ID','Heart_Attack','HyperTension','Heart_Disease'],axis=1)
num_df1 = num_df1.select_dtypes(include=['int64','float64']).copy()
num_df1.head()#num_df1 this df has 3  numerical colms.


# In[63]:


num_df1.shape


# In[64]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_df1 = pd.DataFrame(scaler.fit_transform(num_df1), columns = num_df1.columns)
num_df1#scaling of numerical var in num_df1 var and scaled values stored in same var
#num_df1 had num vars with original values.now it has scaled values of 3 num vars.


# In[65]:


df_pre=pd.concat([num_df1,df1_cat],axis=1)
df_pre.reset_index(inplace=True)#preprocessed df ready for modelling..no patient id and no dep var


# In[66]:


df_pre


# In[67]:


df_pre.dtypes


# In[68]:


plt.figure(figsize=(15,15))

sns.heatmap(df_pre.corr(),annot=True,annot_kws={'size':10})
plt.show()


# In[69]:


df_pre.columns


# #Employment_Type_children and Age have  negative correlation of -0.64.
# #Employment_Type_children and Is_Married_Yes have negative correlation of -0.55.
# #Employment_Type_Self-employed and Employment_Type_Private have negative correlation of -0.5.

# In[70]:


df1.dtypes


# In[71]:


from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[72]:


vif=pd.DataFrame()
vif['VIF']=[variance_inflation_factor(df_pre.values,i) for i in range(df_pre.shape[1])]
vif['Features']=df_pre.columns
vif.sort_values('VIF',ascending=False)


# In[73]:


#No multi collinearity  present amongst  independent variables.


# ## Class Imbalance

# In[74]:


df1['Heart_Attack'].value_counts()


# In[75]:


sns.countplot(df1['Heart_Attack'])
plt.show()


# In[76]:


x=df_pre
y=df1['Heart_Attack']


# In[77]:


y.value_counts()


# In[78]:


from imblearn.over_sampling import SMOTE

smote=SMOTE(random_state=42)

x_sm,y_sm=smote.fit_resample(x,y)


# In[79]:


print('Before Smote independent variable shape',x.shape)
print('Before Smote target variable shape',y.shape)


# In[80]:


print('After Smote independent variable shape',x_sm.shape)
print('After Smote target variable shape',y_sm.shape)


# In[81]:


sns.countplot(y_sm)
plt.show()


# In[82]:


x.shape


# In[83]:


x_sm=pd.DataFrame(x_sm,columns=x.columns)
x_sm.head()


# In[84]:


x_sm.shape


# In[85]:


from sklearn.model_selection import train_test_split
import statsmodels.api as sm


# In[86]:


X = sm.add_constant(x_sm)

X_train, X_test, y_train, y_test = train_test_split(X, y_sm, random_state = 10, test_size = 0.2)

print('X_train', X_train.shape)
print('y_train', y_train.shape)

print('X_test', X_test.shape)
print('y_test', y_test.shape)


# # 3. Logistic Regression

# In[87]:


logreg = sm.Logit(y_train, X_train).fit()

print(logreg.summary())


# **Calculate the AIC (Akaike Information Criterion) value.**
# 

# In[88]:


print('AIC:', logreg.aic)


# We can use the AIC value to compare different models created on the same dataset.

# ### Interpret the odds for each variable 

# In[89]:


df_odds = pd.DataFrame(np.exp(logreg.params), columns= ['Odds']) 

df_odds


# In[90]:


y_pred_prob = logreg.predict(X_test)

y_pred_prob.head()


# Since the target variable can take only two values either 0 or 1. We decide the cut-off of 0.5. i.e. if 'y_pred_prob' is less than 0.5, then consider it to be 0 else consider it to be 1.

# In[91]:


y_pred = [ 0 if x < 0.5 else 1 for x in y_pred_prob]


# In[92]:


y_pred[0:5]


# #### Plot the confusion matrix.

# In[93]:


cm = confusion_matrix(y_test, y_pred)

conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
            linewidths = 0.1, annot_kws = {'size':25})

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.show()


# In[94]:


# performance measures
acc_table = classification_report(y_test, y_pred)

print(acc_table)


# **Interpretation:** 
# 
# From the above output, we can infer that the recall of the positive class is known as `sensitivity` and the recall of the negative class is `specificity`.
# 
# `support` is the number of observations in the corresponding class.
# 
# The `macro average` in the output is obtained by averaging the unweighted mean per label and the `weighted average` is given by averaging the support-weighted mean per label.

# **Kappa score:** It is a measure of inter-rater reliability. For logistic regression, the actual and predicted values of the target variable are the raters.

# In[95]:


kappa = cohen_kappa_score(y_test, y_pred)

print('kappa value:',kappa)


# **Interpretation:** As the kappa score for the full model (with cut-off probability 0.5) is 0.0, we can say that there is substantial agreement between the actual and predicted values.

# #### Plot the ROC curve.
# 
# ROC curve is plotted with the true positive rate (tpr) on the y-axis and false positive rate (fpr) on the x-axis. The area under this curve is used as a measure of separability of the model.

# In[96]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier (Full Model)', fontsize = 15)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)

plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(metrics.roc_auc_score(y_test, y_pred_prob),4)))

plt.grid(True)


# **Interpretation:** The red dotted line represents the ROC curve of a purely random classifier; a good classifier stays as far away from that line as possible (toward the top-left corner).<br>
# From the above plot, we can see that our classifier (logistic regression) is away from the dotted line; with the AUC score 0.8347.

# <a id="cut_off"></a>
# ## 3.1 Identify the Best Cut-off Value

# <a id="cost"></a>
# ###  3.1.1 Cost-based Method

# In[97]:


def calculate_total_cost(actual_value, predicted_value, cost_FN, cost_FP):

    cm = confusion_matrix(actual_value, predicted_value)           
    
    cm_array = np.array(cm)
    
    return cm_array[1,0] * cost_FN + cm_array[0,1] * cost_FP

df_total_cost = pd.DataFrame(columns = ['cut-off', 'total_cost'])

i = 0

for cut_off in range(10, 100):
    total_cost = calculate_total_cost(y_test,  y_pred_prob.map(lambda x: 1 if x > (cut_off/100) else 0), 3.5, 2) 
    df_total_cost.loc[i] = [(cut_off/100), total_cost] 

    i += 1


# In[98]:


df_total_cost.sort_values('total_cost', ascending = True).head(10)


# In[99]:


y_pred_cost = [ 0 if x < 0.7 else 1 for x in y_pred_prob]


# #### Plot the confusion matrix.

# In[100]:


cm = confusion_matrix(y_test, y_pred_cost)

conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
            linewidths = 0.1, annot_kws = {'size':25})

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.show()


# **Compute various performance metrics.**

# In[101]:


acc_table = classification_report(y_test, y_pred_cost)

print(acc_table)


# **Interpretation:** From the above output, we can see that the model with cut-off = 0.7 is 49% accurate. 

# In[102]:


kappa = cohen_kappa_score(y_test, y_pred_cost)

print('kappa value:',kappa)


# <a id="rfe"></a>
# # 4. Recursive Feature Elimination (RFE)

# In the linear regression module, we learn about various techniques for selecting the significant features in the dataset. In this example, let us consider the RFE method for feature selection.

# In[154]:


X_train_rfe = X_train.iloc[:,1:]
X_test_rfe = X_test.iloc[:,1:]

logreg = LogisticRegression()

rfe_model = RFE(estimator = logreg, n_features_to_select = 10)

rfe_model = rfe_model.fit(X_train_rfe, y_train)

feat_index = pd.Series(data = rfe_model.ranking_, index = X_train_rfe.columns)

signi_feat_rfe = feat_index[feat_index==1].index

print(signi_feat_rfe)


# #### Build the logisitc regression model using the variables obtained from RFE.

# In[104]:


logreg_rfe = sm.Logit(y_train, X_train[['Age', 'Employment_Type_children', 'Smoking_Habits_smokes']]).fit()

print(logreg_rfe.summary())


# **Calculate the AIC (Akaike Information Criterion) value.**
# 
# It is a relative measure of model evaluation. It gives a trade-off between model accuracy and model complexity.

# In[105]:


print('AIC:', logreg_rfe.aic)


# **Do predictions on the test set.**

# In[106]:


y_pred_prob_rfe = logreg_rfe.predict(X_test[['Age', 'Employment_Type_children', 'Smoking_Habits_smokes']])

y_pred_prob_rfe.head()


# In[107]:


y_pred_rfe = [ 0 if x < 0.6 else 1 for x in y_pred_prob_rfe]


# In[108]:


y_pred_rfe[0:5]


# #### Plot the confusion matrix.

# In[109]:


cm = confusion_matrix(y_test, y_pred_rfe)

conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
            linewidths = 0.1, annot_kws = {'size':25})

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.show()


# #### Compute the performance measures.

# In[110]:


result = classification_report(y_test, y_pred_rfe)

print(result)


# In[111]:


kappa = cohen_kappa_score(y_test, y_pred_rfe)

print('kappa value:',kappa)


# #### Plot the ROC curve.

# In[112]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier (Full Model)', fontsize = 15)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)

plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(metrics.roc_auc_score(y_test, y_pred_prob),4)))

plt.grid(True)


# ## Decision Tree

# In[160]:


from sklearn.tree import DecisionTreeClassifier  
classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(X_train, y_train) 


# In[161]:


y_pred= classifier.predict(X_test)  


# In[162]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Admission Prediction Classifier (Full Model)', fontsize = 15)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 15)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 15)

plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(metrics.roc_auc_score(y_test, y_pred),4)))

plt.grid(True)


# In[163]:


cm= confusion_matrix(y_test, y_pred)  

conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])

sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = ListedColormap(['lightskyblue']), cbar = False, 
            linewidths = 0.1, annot_kws = {'size':25})

plt.xticks(fontsize = 20)

plt.yticks(fontsize = 20)

plt.show()


# In[ ]:




