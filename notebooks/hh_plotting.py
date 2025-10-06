# Various plotting function for FAO and SHURF projects

import matplotlib.pyplot as plt
import matplotlib.ticker
import pandas as pd
import plotly.express as px

def df_for_plotting(model_name):
    from querysets import countrydata
    filename_descriptives = 'Evaluation/' + model_name + '_descriptives_bcm.parquet'   
    df = pd.read_parquet(filename_descriptives)
    try:
        country_data_df.head()
        df = df.merge(country_data_df, how='left', on=['month_id','country_id'])
    except:
        country_data_df = countrydata()
        df = df.merge(country_data_df, how='left', on=['month_id','country_id'])
    df['Index'] = df.index.values
    return df

def percentiles_over_time_plot_distribution(df, model_name, country_name, step_to_plot, month_to_plot, percentiles_to_plot, outcome, log_outcome, filepath):
        # Your code here
    plt.figure(figsize=(8, 8))
    df = df.loc[df['country_name']=='Sudan']
#    df = df.query(f'country_name == {country_name}').copy()
#    country_name = df.loc[month_to_plot]['country_name'].values[0]
    if country_name == 'Israel':
        country_name = 'Israel-Palestine'
    df.reset_index(inplace=True)
    df.head()
    #colors = ['purple','blue','mediumturquoise','turquoise','green','yellowgreen','yellow','orange','darkorange','red']
    # see https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    colors = ['purple','blueviolet','blue','teal', 'green','yellowgreen','yellow','gold','orange','darkorange','red','purple']
    i=0
    for perc in percentiles_to_plot:
        step_colname = 'step_pred_' + str(step_to_plot)
        perc_colname = str(int(perc*100)) + '%'
        #print(colname)
        if log_outcome:
            plt.plot(df['month_id'],np.expm1(df[step_colname,perc_colname]),label=perc_colname, color = colors[i])
        else:
            plt.plot(df['month_id'],df[step_colname,perc_colname],label=perc_colname, color = colors[i])
        i = i + 1
    alpha = 0.5
    if log_outcome:
        plt.bar(df['month_id'],np.expm1(df[outcome,'mean']),alpha=alpha)
#        plt.semilogy()
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        plt.bar(df['month_id'],df[outcome,'mean'],alpha=alpha)

    plt.legend()
    #locs, labels = plt.xticks()
    #xticks([1,12,24,36,48],['19-1','19-12','20-12','21-12','22-12'])
    plt.xlabel('Month')
    plt.ylabel('Number of fatalities')

#    labels = ['Jan19','','','','','Jun19','','','','','','Dec19','','','','','','Jun20','','','','','','Dec20','','','','','','Jun21','','','','','','Dec21','','','','','','Jun22','','','','','','Dec22']
    plt.xticks(df['month_id'], rotation='vertical')
#    plt.xticks(df['month_id'], labels, rotation='vertical')
    plt.title('Observed and predicted fatalities, ' + model_name + ',' + str(cnt) + ' ' + country_name + ', 2019--22' + ', step ' + str(step_to_plot))
    # Displaying the plot
    #plt.show()
    filename = 'Percentiles_' + model_name + '_s' + str(step_to_plot) + '_' + country_name
    plt.savefig(filepath + filename + '.png')   
    

def percentiles_over_time_plot(df, model_name, cnt, step_to_plot, percentiles_to_plot, outcome, filepath):
        # Your code here
    plt.figure(figsize=(8, 8))
    df = df.query(f'country_id == {cnt}').copy()
    country_name = df.loc[month_to_plot]['country_name'].values[0]
    if country_name == 'Israel':
        country_name = 'Israel-Palestine'
    df.reset_index(inplace=True)
    df.head()
    #colors = ['purple','blue','mediumturquoise','turquoise','green','yellowgreen','yellow','orange','darkorange','red']
    # see https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib
    colors = ['purple','blueviolet','blue','teal', 'green','yellowgreen','yellow','gold','orange','darkorange','red','purple']
    i=0
    for perc in percentiles_to_plot:
        step_colname = 'step_pred_' + str(step_to_plot)
        perc_colname = str(int(perc*100)) + '%'
        #print(colname)
        if log_outcome:
            plt.plot(df['month_id'],np.expm1(df[step_colname,perc_colname]),label=perc_colname, color = colors[i])
        else:
            plt.plot(df['month_id'],df[step_colname,perc_colname],label=perc_colname, color = colors[i])
        i = i + 1
    alpha = 0.5
    if log_outcome:
        plt.bar(df['month_id'],np.expm1(df[outcome,'mean']),alpha=alpha)
#        plt.semilogy()
        plt.gca().yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    else:
        plt.bar(df['month_id'],df[outcome,'mean'],alpha=alpha)

    plt.legend()
    #locs, labels = plt.xticks()
    #xticks([1,12,24,36,48],['19-1','19-12','20-12','21-12','22-12'])
    plt.xlabel('Month')
    plt.ylabel('Number of fatalities')

#    labels = ['Jan19','','','','','Jun19','','','','','','Dec19','','','','','','Jun20','','','','','','Dec20','','','','','','Jun21','','','','','','Dec21','','','','','','Jun22','','','','','','Dec22']
    plt.xticks(df['month_id'], rotation='vertical')
#    plt.xticks(df['month_id'], labels, rotation='vertical')
    plt.title('Observed and predicted fatalities, ' + model_name + ',' + str(cnt) + ' ' + country_name + ', 2019--22' + ', step ' + str(step_to_plot))
    # Displaying the plot
    #plt.show()
    filename = 'Percentiles_' + model_name + '_s' + str(step_to_plot) + '_' + country_name
    plt.savefig(filepath + filename + '.png')   
    
    

def percentile_map(df, log_outcome, step_to_plot, percentile_to_plot, month_to_plot, save_file_path, model_name):
    
    col_to_plot = percentile_to_plot + '_percentile'
    df[col_to_plot] = df[(f'step_pred_{step_to_plot}', percentile_to_plot)]
    if log_outcome:
        range = [0,8]
    else:
        range = [0,2000]
    
    fig = px.choropleth(df.loc[month_to_plot], locations="isoab",
                        color=col_to_plot, 
                        hover_name="country_name", # column to add to hover information
                        hover_data=["Index"],
                        color_continuous_scale=px.colors.sequential.Rainbow,
                        range_color=range,
                        projection="mollweide",
                        title='Predictions, ' + percentile_to_plot + ' percentile, month ' + str(month_to_plot) + ', step ' + str(step_to_plot)
                        )
    fig.update_layout(margin=dict(l=0, r=0, t=25, b=0),                
                    legend=dict(
                        yanchor="bottom",
                        y=0.99,
                        xanchor="left",
                        x=0.01))
    fig.show()
    filename = save_file_path + 'prediction_map_' + model_name + '_s' + str(step_to_plot) + '_perc' + percentile_to_plot + '_m' + str(month_to_plot) + '.html'
    fig.write_html(filename)
    print(f'File saved at {filename}')