# Various utility functions for the uncertainty and FAO projects
import pandas as pd


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from dateutil.relativedelta import relativedelta
import datetime

import xarray as xr
import xskillscore as xs
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def add_onset_indicator(df, onset_def_feature, onset_def_value):
    df['incidence'] = np.where(df['ln_ged_sb_dep']>0,1,0)
    df['onset_risk'] = np.where(df[onset_def_feature]<onset_def_value,1,0)
    df['onset'] = df['incidence'] * df['onset_risk']
    return df


def compute_xr_metrics(predictions_df,steps, outcome, log_outcome, onset_def_feature, onset_def_value):
    
    metrics_list = []
    
    idx = pd.IndexSlice
    print('Computing crps, rmse, mse, brier scores')
    mean_predictions_df = pd.DataFrame(predictions_df.groupby(level=[0, 1]).mean())
#    predictions_df = add_onset_indicator(predictions_df, onset_def_feature, onset_def_value)
    observations_xr = predictions_df.loc[:,:,0][outcome].to_xarray()
#    observations_onset_xr = predictions_df.loc[:,:,0].loc['onset_risk'==1].to_xarray()
    
    for step in steps:
        step_dict = {
            'step': step,
        }
        forecasts_ensemble_xr = predictions_df['step_pred_' + str(step)].to_xarray()
#        forecasts_ensemble_onset_xr = predictions_df.loc['onset_risk'==1]['step_pred_' + str(step)].to_xarray()
        step_dict['crps'] = xs.crps_ensemble(observations_xr,forecasts_ensemble_xr,member_dim='draw').item()
#        step_dict['crps_onset'] = xs.crps_ensemble(observations_onset_xr,forecasts_ensemble_onset_xr,member_dim='draw').item()
        step_dict['crps_m'] = xs.crps_ensemble(observations_xr,forecasts_ensemble_xr,member_dim='draw',dim='country_id').values
        mean_predictions_xr = mean_predictions_df['step_pred_' + str(step)].to_xarray()
        step_dict['mse'] = xs.mse(observations_xr,mean_predictions_xr).values
        step_dict['rmse'] = np.sqrt(step_dict['mse'])
        for thr in [0.5, 9.5, 99, 999]:
            lnthr = np.log1p(thr)
            step_dict['brier' +  str(np.int32(np.round(thr+.9)))] = xs.threshold_brier_score(observations_xr,forecasts_ensemble_xr,lnthr,member_dim='draw',dim=None).values.mean() 
        metrics_list.append(step_dict)
    return metrics_list
    


def percentile_table(df,filename, colnames,caption, label):
    ''' Writes a table of percentiles to a latex file in a format suitable for the FAO project
    colnames: list of column names to include in the table
    '''
    percentiles = [0.05, 0.5, 0.8, 0.9, 0.95, 0.96, 0.975, 0.98, 0.99, 0.995]
    header = ['Percentile'] + colnames + ['Return period']
    output_df = df.describe(percentiles=percentiles)
    output_df.reset_index(inplace=True)
    output_df.rename(columns={'index':'Percentile'}, inplace=True)
    output_df['Return_period'] = output_df['Percentile'].apply(lambda x: '2' if x == '50%' else '10' if x == '90%' else '20' if x == '95%' else '25' if x == '96%' else '40' if x == '97.5%' else '100' if x == '99%' else '200' if x == '99.5%' else '5' if x == '80%' else '50' if x == '98%' else 'N/A')
    output_df_raw = output_df.copy()
    output_df = output_df.loc[output_df['Return_period'] != 'N/A']
    #output_df.style.format("'pc_ged_sb_next_12': {:.0%}")
    output_df.to_latex(filename, index=False,float_format="{:0.1f}".format, header=header,column_format='l|p{.2\linewidth}|p{.3\linewidth}|r', caption=caption,label=label)
    print('Table written to ', filename)
    return output_df_raw  

     
def all_steps_are_included(df, steps_to_evaluate, prefix='step_pred_'):
    all_included = True
    for step in steps_to_evaluate:
        colname = prefix + str(step)
        if colname not in df.columns:
            print('step not in dataframe')
            all_included = False
            return all_included
    return all_included

def compute_interval_score(model,model_predictions_df,steps,prediction_interval_level):

    model['iscore_across'] = 0
    for step in steps:
        model[f'iscore_step_{step}'] =  interval_score(model_predictions_df['ln_ged_sb_dep'],model_predictions_df[f'step_pred_{step}'],prediction_interval_level).mean()
        model['iscore_across'] = model['iscore_across'] + model[f'iscore_step_{step}']
    model['iscore_across'] = model['iscore_across']/len(steps)

def compute_coverage(model,steps,df):
    model['covered_across'] = 0
    model['covered_nonzeros_across'] = 0
    for step in steps:
        df['observed'] = df['ln_ged_sb_dep']['mean']
        df[f'step_{step}_p10'] = df[f'step_pred_{step}']['10%']
        df[f'step_{step}_p90'] = df[f'step_pred_{step}']['90%']
        df[f'covered_step_{step}'] = 1
        df.loc[(df[f'step_{step}_p10']>df['observed']) | (df[f'step_{step}_p90']<df['observed']), f'covered_step_{step}'] = 0
        model[f'covered_step_{step}'] = df[f'covered_step_{step}'].mean()
        threshold = 0   
        model[f'covered_nonzeros_step_{step}'] = df.loc[df['observed']>np.log1p(threshold)][f'covered_step_{step}'].mean()
        model['covered_across'] = model['covered_across'] + model[f'covered_step_{step}']
        model['covered_nonzeros_across'] = model['covered_nonzeros_across'] + model[f'covered_nonzeros_step_{step}']
    model['covered_across'] = model['covered_across']/len(steps)
    model['covered_nonzeros_across'] = model['covered_nonzeros_across']/len(steps)
    


def describe_model(model,percentiles_in_describe, steps_to_evaluate, outcome, delete_predictions,rerun_all):

    if delete_predictions:
        model['predictions'] = pd.read_parquet('Predictions/' + model['model_name'] + '.parquet')
    
    def compute_main_descriptives():
        model['descriptives'] = model['predictions'][[outcome] + steps_to_evaluate].describe(percentiles=percentiles_in_describe)
        model['filename_descriptives'] = 'Evaluation/' + model['model_name'] + '_descriptives.parquet'
        print('Recomputing main descriptives. Saving descriptives to',model['filename_descriptives'])
        model['descriptives'].to_parquet(model['filename_descriptives'])
        
    def compute_descriptives_by_month():
        model['descriptives_by_month'] = model['predictions'].groupby(level=[0]).describe(percentiles=percentiles_in_describe)
        model['filename_descriptives_bm'] = 'Evaluation/' + model['model_name'] + '_descriptives_bm.parquet'
        model['descriptives_by_month'].to_parquet(model['filename_descriptives_bm'])
        print('Recomputing descriptives by month. Saving to ',model['filename_descriptives_bm'])
    
    def compute_descriptives_by_country():
        model['filename_descriptives_bc'] = 'Evaluation/' + model['model_name'] + '_descriptives_bc.parquet'
        print('Recomputing descriptives by country. Saving to ',model['filename_descriptives_bm'],' and ',model['filename_descriptives_bc'])
        model['descriptives_by_country'] = model['predictions'].groupby(level=[1]).describe(percentiles=percentiles_in_describe)
        model['descriptives_by_country'].to_parquet(model['filename_descriptives_bc'])
        
        
    def compute_descriptives_by_cm():
        model['filename_descriptives_bcm'] = 'Evaluation/' + model['model_name'] + '_descriptives_bcm.parquet'
        print('Recomputing descriptives by country month. Saving to ',model['filename_descriptives_bcm'])
        model['descriptives_by_cm'] = model['predictions'].groupby(level=[0,1]).describe(percentiles=percentiles_in_describe)
        model['descriptives_by_cm'].to_parquet(model['filename_descriptives_bcm'])
        
    
    if rerun_all: 
        print('rerun_all=True. Rerunning')
        compute_main_descriptives()
        compute_descriptives_by_month()
        compute_descriptives_by_country()
        compute_descriptives_by_cm()
    if not rerun_all:
        try:
            model['filename_descriptives'] = 'Evaluation/' + model['model_name'] + '_descriptives.parquet'
            print('rerun_all=False. Reading in',model['filename_descriptives'])
            model['descriptives'] = pd.read_parquet(model['filename_descriptives'])
            if not all_steps_are_included(model['descriptives'], steps_to_evaluate, prefix=''):
                print('Not all steps are included in the main descriptives file. Recomputing.')
                compute_main_descriptives()
        except:
            print('Exception')
            compute_main_descriptives()
        try:
            model['filename_descriptives_bm'] = 'Evaluation/' + model['model_name'] + '_descriptives_bm.parquet'
            print('Reading in',model['filename_descriptives_bm'])
            model['descriptives_by_month'] = pd.read_parquet(model['filename_descriptives_bm'])
            if not all_steps_are_included(model['filename_descriptives_bm'], steps_to_evaluate, prefix=''):
                print('Not all steps are included in the by month descriptives file. Recomputing.')
                compute_descriptives_by_month()
        except:
            print('Exception')
            compute_descriptives_by_month()
            
        try:
            model['filename_descriptives_bc'] = 'Evaluation/' + model['model_name'] + '_descriptives_bc.parquet'
            print('Reading in',model['filename_descriptives_bc'])
            model['descriptives_by_country'] = pd.read_parquet(model['filename_descriptives_bc'])
            if not all_steps_are_included(model['filename_descriptives_bc'], steps_to_evaluate, prefix=''):
                print('Not all steps are included in the by country descriptives file. Recomputing.')
                compute_descriptives_by_country()
        except:
            print('Exception')
            compute_descriptives_by_country()
            
        try:
            model['filename_descriptives_bcm'] = 'Evaluation/' + model['model_name'] + '_descriptives_bcm.parquet'
            print('Reading in',model['filename_descriptives_bcm'])
            model['descriptives_by_cm'] = pd.read_parquet(model['filename_descriptives_bcm'])
            if not all_steps_are_included(model['filename_descriptives_bcm'], steps_to_evaluate, prefix=''):
                print('Not all steps are included in the by country-month descriptives file. Recomputing.')
                compute_descriptives_by_cm()
        except:
            print('Exception')
            compute_descriptives_by_cm()
    
    if delete_predictions:
        model['predictions'] = []   
        model['descriptives_by_month'] = []
        model['descriptives_by_country'] = []
        model['descriptives_by_cm'] = []



def interval_score(observed: np.array, predictions: np.array, prediction_interval_level: float = 0.95) -> np.array:
    """
    Written by Jonas Vestby for VIEWS prediction competition
    Interval Score implemented based on the scaled Mean Interval Score in the R tsRNN package https://rdrr.io/github/thfuchs/tsRNN/src/R/metrics_dist.R

    The Interval Score is a probabilistic prediction evaluation metric that weights between the narrowness of the forecast range and the ability to correctly hit the observed value within that interval.
    
    :param observed: observed values
    :type observed: array_like
    :param predictions: probabilistic predictions with the latter axis (-1) being the forecasts for each observed value
    :type predictions: array_like
    :param prediction_interval_level: prediction interval between [0, 1]
    :type prediction_interval_level: float
    :returns array_like with the interval score for each observed value
    :rtype array_like

    observed = np.random.negative_binomial(5, 0.8, size = 600)
    forecasts = np.random.negative_binomial(5, 0.8, size = (600, 1000))

    score = interval_score(observed, forecasts)
    print(f'MIS: {score.mean()}')

    """

#    assert 0 < prediction_interval_level < 1, f"'prediction_interval_level' must be a number between 0 and 1." 

    alpha = 1 - prediction_interval_level
    lower = np.quantile(predictions, q = alpha/2, axis = -1)
    upper = np.quantile(predictions, q = 1 - (alpha/2), axis = -1)

    interval_width = upper - lower
    lower_coverage = (2/alpha)*(lower-observed) * (observed<lower)
    upper_coverage = (2/alpha)*(observed-upper) * (observed>upper)

    return(interval_width + lower_coverage + upper_coverage)


def percentile_table(df,filename, colnames,caption, label):
    ''' Writes a table of percentiles to a latex file in a format suitable for the FAO project
    colnames: list of column names to include in the table
    '''
    percentiles = [0.05, 0.5, 0.8, 0.9, 0.95, 0.96, 0.975, 0.98, 0.99, 0.995]
    header = ['Percentile'] + colnames + ['Return period']
    output_df = df.describe(percentiles=percentiles)
    output_df.reset_index(inplace=True)
    output_df.rename(columns={'index':'Percentile'}, inplace=True)
    output_df['Return_period'] = output_df['Percentile'].apply(lambda x: '2' if x == '50%' else '10' if x == '90%' else '20' if x == '95%' else '25' if x == '96%' else '40' if x == '97.5%' else '100' if x == '99%' else '200' if x == '99.5%' else '5' if x == '80%' else '50' if x == '98%' else 'N/A')
    output_df_raw = output_df.copy()
    output_df = output_df.loc[output_df['Return_period'] != 'N/A']
    #output_df.style.format("'pc_ged_sb_next_12': {:.0%}")
    output_df.to_latex(filename, index=False,float_format="{:0.1f}".format, header=header,column_format='l|p{.3\linewidth}|p{.3\linewidth}|r', caption=caption,label=label)
    return output_df_raw  


def add_cumulative_coming_months(df, basis_col, output_col, steps, log=False, pre=True):
    ''' Adds a column to the dataframe that contains the cumulative sum of the basis_col 
    for the next steps number of months.
    
    log: if True, the basis_col is log transformed before the cumulative sum is calculated.
    pre: if True, the cumulative sum is calculated for the previous steps number of months. 
         if False, it is calculated for the next steps number of months.
    '''
    if log:
        df['temp_col'] = np.expm1(df[basis_col])
    else:
        df['temp_col'] = df[basis_col] 
    df['rolling_sum_post'] = df.groupby(['country_name'])['temp_col'].transform(lambda x: x.rolling(steps, steps).sum())
    df['rolling_sum_pre'] = df.groupby(['country_name'])['rolling_sum_post'].shift(-(steps-1))
    if pre:
        df[output_col] = df['rolling_sum_pre']
    else:
        df[output_col] = df['rolling_sum_post']
    if log:
        df[output_col] = np.log1p(df[output_col])
    else:
        df[output_col] = df[output_col]
    df.drop(columns=['temp_col','rolling_sum_post','rolling_sum_pre'], inplace=True)
    
    return df

# Adding cumulative forecasts
def add_cumulative_forecasts(df, steps, log=False):
    ''' Adds cumulative forecasts for the specified number of steps to the dataframe.
        Assumes that the dataframe has a 'country_name' column, a 'step_pred_{step}' column 
        and a 'pc_step_pred_{step}' column for each step to go into the cumulation.
        '''
    
    df['cumulative_forecasts_log_'+str(steps)] = 0
    df['cumulative_forecasts_nonlog_'+str(steps)] = 0
    df['cumulative_forecasts_pc_'+str(steps)] = 0
    if log:
        for this_step in range(1, steps+1):
            df['temp_col'] = np.expm1(df[f'step_pred_{this_step}'])
            df['cumulative_forecasts_'+str(steps)] += df.groupby(['country_name'])['temp_col'].shift(-(this_step-1))
        df['cumulative_forecasts_nonlog_'+str(steps)] = df['cumulative_forecasts_'+str(steps)]
        df['cumulative_forecasts_log_'+str(steps)] = np.log1p(df['cumulative_forecasts_'+str(steps)])
    else:
        for this_step in range(1, steps+1):
            df['cumulative_forecasts_nonlog_'+str(steps)] += df.groupby(['country_name'])[f'step_pred_{this_step}'].shift(-(this_step-1))

    return df


def add_month_year(df):
    ''' Adds month and year columns to the dataframe, based on the month_id column.'''
    df.reset_index(inplace=True)    
    df['month'] = df['month_id'] % 12
    df['month'][df['month'] == 0] = 12
    df['year'] = 1980 + (df['month_id'] // 12)
    df['year'][df['month'] == 12] = df['year'] - 1
    df.set_index(['month_id', 'country_id'], inplace=True)
    return df    



def month_id_to_date(month_id: int) -> str:
    """Converts a month_id to a string with the month and year.
    Parameters
    ----------
    month_id : int
        A count of months starting (from 1) on January 1980.
    Returns
    -------
    str
        The month and year of the month_id in "Month Year" format.
    """
    start_date = datetime.date(1980, 1, 1)
    target_date = start_date + relativedelta(months=month_id-1)
    return target_date.strftime("%B %Y")

def plot_density_yearly_cm(year_file,country_id,save_file_path,year):
    
    # Create a figure with two subplots side by side
    fig = plt.figure(figsize=(15, 6))
    # Define the grid
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.2])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    
    cm_yearly_df = pd.read_parquet(year_file)
    cm_yearly_df.reset_index(inplace=True)
    cm_yearly_df_country = {}
    for i in range(1, 247):
        cm_yearly_df_country[i] = cm_yearly_df[cm_yearly_df['country_id'] == i]
    df = cm_yearly_df_country[country_id]
    
    percentiles = df.groupby('month_id')['outcome'].quantile(
        [0.2, 0.5, 0.8]).unstack()
    # Plot the heat plot on the first subplot
    sns.kdeplot(
        data=df, x="month_id", y="outcome",
        fill=True, common_norm=True, palette="crest",
        alpha=.5, linewidth=0, cmap='rainbow', levels=50, thresh=0, cbar=False, ax=ax1
    )
    # Add trend lines for each percentile
    percentiles[0.2].plot(ax=ax1, style='--', color='black',
                        label='20th percentile')
    percentiles[0.5].plot(ax=ax1, style='-', color='black',
                        label='50th percentile')
    percentiles[0.8].plot(ax=ax1, style='-.', color='black',
                        label='80th percentile')
    ax1.set_xlim(df['month_id'].min(), df['month_id'].max())
    ax1.set_ylim(df['outcome'].min(), df['outcome'].max())
    # Change x-axis labels to month names
    x_ticks = range(df['month_id'].min(), df['month_id'].max() + 1)
    x_labels = [month_id_to_date(month_id) for month_id in x_ticks]
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels, rotation=90)
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Outcome")
    ax1.set_title("Density Plot of Outcome by Month with Percentile Trend Lines")
    # Add the legend
    legend = ax1.legend()
    plt.setp(legend.get_texts(), color='black')  # Set legend text color to black
    # Change legend background to white
    legend.get_frame().set_facecolor('white')
    # Plot the first KDE on the second subplot without legend
    sns.kdeplot(
        data=df, y="outcome", hue="month_id",
        fill=True, common_norm=False, palette="crest",
        alpha=.5, linewidth=1, ax=ax2, legend=False
    )
    ax2.set_ylim(df['outcome'].min(
    ), df['outcome'].max())
    plt.tight_layout()
    plt.savefig(save_file_path + f'heatplot_{country_id}_{year}.png')
    plt.show()