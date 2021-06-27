## site from where this code is used
## https://stackoverflow.com/questions/57531388/how-can-i-reduce-the-memory-of-a-pandas-dataframe

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tqdm
from scipy import stats
from datetime import datetime
import calendar
import time
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
import datetime


## Global Lambda functions
def selectLambdaFunc(lambda_func):
    if lambda_func == 'mode':
        return lambda x : stats.mode(x)[0][0]
    elif lambda_func == 'nunique':
        return lambda x : x.nunique()
    elif lambda_func == 'count':
        return lambda x : len(x)
    elif lambda_func == 'min':
        return lambda x : x.value_counts().index[-1]
    elif lambda_func == 'max':
        return lambda x : x.value_counts().index[0]
    else:
        return lambda x : x



def reduce_mem_usage(df, verbose=True):
    """
        In this function the size of dataframe is reduced by checking the size of data contained in each series
        and seeing if minimum or maximum size of the element present in the series is well within the limits of 
        the new representation that it will be assigned.
        For eg: if a series contains 'int' type of data
                then is the size of maximum and minimum element 
                within int-8 representation if yes then we change
                the type to int-8
                else we go further and check for other sizes of integer
                i.e int-16,int-32,int-64 and so on.
    """
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
def computeCategoricalAggr(df,aggr_cat_col,lambda_func):
    """
    Input:
        df  : dataframe to be worked upon
aggr_cat_col: column to be aggregated
 lambda_func: lambda function to be applied
 col_suffix : suffix to be appended for column name
   Output:
       returns dataframe with categorical variable value and it's corresponding aggregate function value
    """
    cat_df = df.groupby('card_id')[aggr_cat_col].apply(selectLambdaFunc(lambda_func)).reset_index().rename(columns={aggr_cat_col:aggr_cat_col + '_' + lambda_func})
    # merge_df = pd.merge(nout_train_df,cat_df,on='card_id',how='inner')
    return cat_df
    

def generateAggrColumns(df,groupby_col,aggr_col,aggr_funcs,col_prefix,isCardId):
    """
        Input: merge_df : df to be merged with
                     df : df whose columns will be used for groupby and aggregation
            groupby_col : column to be used for groupby
               aggr_col : column to be used for aggregation
               aggr_func: list of aggregate functions to be applied
             col_prefix : prefix to be applied for newly generated column name
               isCardId : if True then 'card_id' is used for groupby
               
        Output merge_df : merged df with aggregate columns
        
            Description : for each categorical value in categorical column we create a 
                          aggregate column with provided aggregate functions
                          for instance : category_1 contains two categories 'Y' and 'N'
                          and suppose aggregate function list contains min,max,median
                          then aggregate columns cat1_y_min,cat1_y_max,cat1_y_median
                          cat1_n_min,cat1_n_max,cat1_n_median will be created using groupby 
                          columns.
    """
    groupby = [groupby_col]
    if isCardId:
        groupby = ['card_id'] + [groupby_col]
    
    firstRun = True
    merge_df = pd.DataFrame()
    aggr_dict = {aggr_col : aggr_funcs}
    
    for cat in tqdm(df[groupby_col].unique()):
        print("Category of Column : ",cat)
        aggr_prefix = {aggr_col : col_prefix + '_' + str(cat)}
        
        temp_df = df[df[groupby_col]==cat].groupby(groupby).agg(aggr_dict).reset_index().rename(columns=aggr_prefix)
        temp_df.columns = ['_'.join(col).strip('_') for col in temp_df.columns.values]
        temp_df.drop([groupby_col],axis = 1,inplace = True)
        
        if firstRun:
            merge_df = temp_df
            firstRun = False
        else:
            merge_df = pd.merge(merge_df,temp_df,on='card_id',how='outer')
            print("***********Merge_df**********")
            print(merge_df.shape)
    return merge_df


def removeMissingValuesColumns(df,missing_perc=0.5):
    """
              df : dataframe to be considered for removing missing values
    missing_perc : filter to decide columns which should be removed with 
                   percentage of missing values greater than the passed value  
    """
    null_val_col = (df.isnull().sum()/df.shape[0] >= missing_perc).reset_index().rename(columns= {'index' : 'cols',0 : 'bool'})
    cols = null_val_col[null_val_col['bool'] == False]['cols'].values
    fig,ax = plt.subplots(figsize = (15,10))
    sns.heatmap(abs(df[cols].corr()),ax=ax)
    return df[cols]
    
    
## Let's write a generic function to display barplots
def plotBarPlot(df,col,quantile=0):
    """
          df: dataframe which contains the column to be aggregated
         col: column which is used for counting the number of categories
    quantile: used as a filter to display the number of categories starting from count of 0th quantile as default  
    """
    qq = np.percentile(df[col].value_counts().values,q=quantile)
    id_counts = df[col].value_counts().reset_index().rename(columns = {col : 'count'}).sort_values(by=['count'])
    filtered_data = id_counts[id_counts['count'] >= qq]
    
    #Plotting the barplot
    fig,ax = plt.subplots(figsize = (15,10))
    sns.barplot('index','count',data = filtered_data,ax=ax)
    plt.xlabel(col)
    plt.xticks(rotation = 90)
    
    
    
## Creating sampled_df with historical transactions for 100 cardId's
def createSampledHistDf(temp_df,card_ids,sortby):
    """
        Sampling historical transactions based on 
        card_ids recieved in input parameter
        and sorting them by given list of columns
    """
    sampled_card_ids = temp_df[['card_id']].sample(n=100)
    sampled_df = temp_df[temp_df['card_id'].isin(card_ids)].copy()
    sampled_df.sort_values(by = sortby,inplace = True)
    return sampled_df
    
    
## Picking train_df rows which contain this cardId's
def createSampledTrainDf(temp_df,card_ids):
    """
        Sampling train_df based on card_ids
    """
    s_train_df = temp_df[temp_df['card_id'].isin(card_ids)].copy()
    return s_train_df


def createPurchaseAmountLagFeatures(sampled_df,aggr_funcs):
    """
        Creating dataframe for data grouped by card_id and month_lag,aggregating over new_purchase_amounts 
    """
    print("Creating purchase_amount aggregation wrt month_lags...")
    aggr_dict = {'new_purchase_amount' : aggr_funcs}
    aggr_df = sampled_df.groupby(['card_id','month_lag']).agg(aggr_dict).reset_index().rename(columns = {'new_purchase_amount' : 'np_amount'})
    aggr_df.columns = ['_'.join(col).strip('_') for col in aggr_df.columns.values]
    
    #List to create column month_lag
    month_lag = aggr_df['month_lag'].unique()
    # creating list of column names with np_amount aggregated features
    np_amount_cols = ['np_amount_' + func for func in aggr_funcs] 

    first_run = True #Used as a boolean to decide if dataframe has to be merged in first instance or not
    merge_df = pd.DataFrame() #For merging columns with different aggregate functions on purchase amount with month lag information
    
    print("Merging aggregate columns...")
    ## Create lag features for different aggregated np_amount
    for col in tqdm(np_amount_cols):
        for lag in tqdm(month_lag):
            temp_lag_df = aggr_df[aggr_df['month_lag'] == lag][['card_id',col]].rename(columns={col : col + "_" + str(lag)})
            
            if first_run:
                merge_df = temp_lag_df
                first_run = False
            else:
                merge_df = pd.merge(merge_df,temp_lag_df,on='card_id',how='outer')
    
    return merge_df


def createPurchaseAmountRatioFeatures(df,aggr_funcs,lag_shift,month_lag):
    np_amount_cols = ['np_amount_' + func for func in aggr_funcs] 
    max_month_lag = month_lag.max()
    min_month_lag = month_lag.min()
    
    final_df = pd.DataFrame() #For creating final dataframe with ratio of purchase amount features obtained from above dataframe
    ## Creating ratios of these lag features 
    for col in tqdm(np_amount_cols):
        for lag in tqdm(range(min_month_lag,max_month_lag-lag_shift+1)):
            future_col = col + '_'+ str(lag+lag_shift)
            past_col = col + '_' + str(lag)
            final_df[col + '_' + str(lag) + '_' + str(lag+lag_shift)] = df[future_col]/df[past_col]
            
    final_df['card_id'] = df['card_id']
    ## Creating time
    return final_df


## Creating simple window averaging over previous month lags
def createPurchaseAmountWindowAvg(df,aggr_funcs,lag_shift,month_lag):
    np_amount_cols = ['np_amount_' + func for func in aggr_funcs] 
    max_month_lag = month_lag.max()
    min_month_lag = month_lag.min()
    
    #For creating final dataframe with window average of purchase amount features obtained from input dataframe
    final_df = pd.DataFrame() 
    ## Calculating simple window average of these lag features 
    for col in tqdm(np_amount_cols):
        for lag in tqdm(range(min_month_lag,max_month_lag-lag_shift+1)):
            
            future_col = lag+lag_shift
            past_col = lag
            avg = 0
            
            for i in range(past_col,future_col):
                avg = avg + df[col + '_'+ str(i)]
            avg = avg/(lag_shift+1)    
            final_df[col + '_' + str(lag) + '_' + str(lag+lag_shift) + '_avg'] = avg

    final_df['card_id'] = df['card_id']
    return final_df
    
    
def createDateRelatedBoolFeat(temp_df):
    start_time = time.clock()
    temp_df['is_purchase_month_end'] = np.full(len(temp_df),False)
    temp_df['is_purchase_month_start'] = np.full(len(temp_df),False)
    temp_df['is_purchase_quarter_start'] = np.full(len(temp_df),False)
    temp_df['is_purchase_quarter_end'] = np.full(len(temp_df),False)
    temp_df['is_purchase_year_end'] = np.full(len(temp_df),False)
    temp_df['is_purchase_year_start'] = np.full(len(temp_df),False)

    temp_df['is_christmas'] = np.full(len(temp_df),False)
    temp_df['is_mothers_day'] = np.full(len(temp_df),False)
    temp_df['is_childrens_day'] = np.full(len(temp_df),False)
    temp_df['is_valentines_day'] = np.full(len(temp_df),False)
    temp_df['is_fathers_day'] = np.full(len(temp_df),False)


    month_end_index = (((temp_df.day == 31) & (temp_df.month.isin([1,3,5,7,8,10,12]))) | 
                       ((temp_df.day == 30) & (temp_df.month.isin([4,6,9,11]))) | 
                       ((temp_df.day == 28) & (temp_df.month == 2)))

    month_start_index = (temp_df.day == 1)
    quarter_start_index = ((temp_df.day == 1) & (temp_df.month.isin([1,4,7,10])))
    quarter_end_index = (((temp_df.day == 31) & (temp_df.month.isin([1,7,10]))) | 
                         ((temp_df.day == 30) & (temp_df.month == 4)))

    year_start_index = (((temp_df.day == 1) & (temp_df.year == 2017)) | 
                        ((temp_df.day == 1) & (temp_df.year == 2018)))

    year_end_index = ((temp_df.day == 31) & (temp_df.year == 2017))

    christmas_day_index = (temp_df.purchase_date == datetime.datetime(2017,12,25))
    mothers_day_index = (temp_df.purchase_date == datetime.datetime(2017,5,14))
    childrens_day_index = (temp_df.purchase_date == datetime.datetime(2017,11,24))
    valentines_day_index = (temp_df.purchase_date == datetime.datetime(2017,2,14))
    fathers_day_index = (temp_df.purchase_date == datetime.datetime(2017,6,18))


    temp_df.loc[month_end_index,'is_purchase_month_end'] = True
    temp_df.loc[month_start_index,'is_purchase_month_start'] = True
    temp_df.loc[quarter_start_index,'is_purchase_quarter_start'] = True
    temp_df.loc[quarter_end_index,'is_purchase_quarter_end'] = True
    temp_df.loc[year_start_index,'is_purchase_year_start'] = True
    temp_df.loc[year_end_index,'is_purchase_year_end'] = True

    temp_df.loc[christmas_day_index,'is_christmas'] = True
    temp_df.loc[mothers_day_index,'is_mothers_day'] = True
    temp_df.loc[childrens_day_index,'is_childrens_day'] = True
    temp_df.loc[valentines_day_index,'is_valentines_day'] = True
    temp_df.loc[fathers_day_index,'is_fathers_day'] = True
    
    end_time = time.clock()
    print("Time taken for completion : ",end_time-start_time)
    return temp_df
    
def createNumericalAggr(df,groupby_cat,num_feat):
    """
        Input : 
            df : DataFrame to be used for operation
    groupby_cat: Column to be used for groupby
       num_feat: Feature across which aggregation will be performed 
    
    returns aggregated features with min,max,median,sum and standard-deviation
    """
    
    aggr_funcs = ['min','max','median','sum','std','skew']
    aggr_dict = {num_feat : aggr_funcs}
    aggr_df = df.groupby([groupby_cat]).agg(aggr_dict).reset_index()
    aggr_df.columns = ['_'.join(col).strip('_') for col in aggr_df.columns.values]
    return aggr_df
    
def createSvdFeatures(df,groupby_col,merch_col,n_comp):
    start_time = time.clock()
    temp_df = df.groupby([groupby_col])[merch_col].apply(lambda x : list(x)).reset_index().rename(columns={merch_col : 'list'})
    temp_df['list'] = temp_df['list'].apply(lambda x : ' '.join([str(element) for element in x]))
    tfidf_vectorizer = TfidfVectorizer()
    print("Performing tfidf vectorization on ",merch_col,"...")
    
    temp_tfidf = tfidf_vectorizer.fit_transform(temp_df['list'])
    svd = TruncatedSVD(n_components = n_comp,random_state = 42)
    svd_temp = svd.fit_transform(temp_tfidf)
    print("Explained Variance ration with ",n_comp," components : ",np.sum(svd.explained_variance_ratio_)*100)
    svd_temp_df = pd.DataFrame(svd_temp)
    svd_temp_df.columns = ['svd_' + str(col) for col in svd_temp_df.columns]
    merge_temp_df = pd.concat([temp_df,svd_temp_df],axis=1)
    end_time = time.clock()
    print("Time taken for completion : ",start_time-end_time)
    return merge_temp_df,tfidf_vectorizer,svd
  
def createWord2VecFeatures(df,groupby_col,cat_col,dim):
    start_time = time.clock()
    df[cat_col] = df[cat_col].astype(str)
    temp_df = df.groupby([groupby_col])[cat_col].apply(lambda x : set(x)).reset_index().rename(columns={cat_col : 'list'})
    w2v_model = Word2Vec(temp_df['list'],size = dim,min_count = 1)
    cat_w2v_matrix = []
    for cat_list in temp_df['list'].values:
        cat_w2v = np.zeros(dim)
        for cat in cat_list:
            cat_w2v += w2v_model.wv[cat]
        cat_w2v = cat_w2v/len(cat_list)
        cat_w2v_matrix.append(cat_w2v)

    temp_df.drop(['list'],axis=1,inplace = True)
    cat_w2v_df = pd.DataFrame(cat_w2v_matrix,columns = ['w2v_'+ str(i) for i in range(dim)])
    temp_df = pd.concat([temp_df,cat_w2v_df],axis=1)
    end_time = time.clock()
    print("Time taken for completion : ",start_time-end_time)
    return temp_df,w2v_model
 
