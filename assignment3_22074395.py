import cluster_tools as clst
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import numpy as np
import scipy.optimize as opt

def load_data(dataset, country_list):
    """
    This function takes in a csv file and a list of countries that are of
    interest. Returns two dataframes one with the years as columns and the
    other country names as columns.

    Parameters
    ----------
    dataset : .csv file
    country_list : List
    """
    # skipping first 4 rows, as they contain non essential data.
    world_bank_df = pd.read_csv(dataset, skiprows=4)

    # Removing non essential data.
    world_bank_df.drop(['Country Code', 'Indicator Code', 'Unnamed: 67'],
                       axis=1, inplace=True)

    # subsetting the dataframe to get data for countries we are interested in.
    world_bank_df = world_bank_df[
        world_bank_df['Country Name'].isin(country_list)]

    # Setting index before transposing the dataframe
    temp_df = world_bank_df.set_index('Country Name')

    return world_bank_df, temp_df.T

def subset_data(data,
                indicator1,indicator2,indicator3,indicator4,
                countries):
    
    # Subsetting the dataframe based on the country names passed
    #data = data[(data['Country Name'] == country1) |
    #            (data['Country Name'] == country2) |
    #            (data['Country Name'] == country3) |
    #            (data['Country Name'] == country4) |
    #            (data['Country Name'] == country5) |
    #            (data['Country Name'] == country6)
    #            ]

    # Selecting only the indicators needed from the dataset.
    data = data[
        (data['Indicator Name'] == indicator1) |
        (data['Indicator Name'] == indicator2) |
        (data['Indicator Name'] == indicator3) |
        (data['Indicator Name'] == indicator4)
    ]      
    result_df = pd.DataFrame()
    for country in countries:
        temp = data[data['Country Name'] == country]
        
        # Dropping Country Name column to make it easier to label
        # and the columns 1960 and 2022 have Na values
        temp = temp.drop(['Country Name', '1960', '2022'], axis=1)
        temp = temp.set_index('Indicator Name')
        temp = temp.T
        temp['Country'] = country
        result_df = pd.concat([result_df, temp], axis=0, ignore_index=True)
        
        
        
    
    return data, temp, result_df

def kmeans_cluster(ncluster, normalised_df, indicator1,indicator2):
    
    plt.figure()
    
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    
    kmeans.fit(normalised_df)
    
    labels = kmeans.labels_
        
    cen = kmeans.cluster_centers_
    
    xkmeans = cen[:,0]
    ykmeans = cen[:,1]
        
    cm = plt.colormaps["Paired"]
    
    plt.scatter(normalised_df[indicator1],
                normalised_df[indicator2], marker="o", c=labels,
                cmap=cm)
    plt.scatter(xkmeans, ykmeans, marker="d", label="kmeans centres", c='black' )
    
    plt.legend()
    
    return cen, labels, cm


def sil_score(normalised_df):
    
    for n in range(2,8):    

        ncluster = n
        
        kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
        
        kmeans.fit(normalised_df)
        
        labels = kmeans.labels_
                
        print("Nclusters: "+str(n))
        print(skmet.silhouette_score(normalised_df, labels))
        
        
def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t)) 
    
    return f

def funct(t,s,k):
    
    x = s * (np.exp(k*t))
    
    return x


def generate_line_plot(data, country, indicator, xlabel, ylabel, title):
    """
    This function generates line plots for the given dataframe for a particular
    indicator in the world bank dataset. It also requires a list of country
    names.
    Parameters
    ----------
    data : TYPE
    countries : TYPE
    indicator : TYPE
    xlabel : TYPE
    ylabel : TYPE
    title : TYPE

    Returns None.

    """
    # Specifying figure size, as plot is big
    #plt.figure(figsize=(15, 8))

    temp_df = data[country].T
    # Subsetting the transposed df. Which now has years as columns
    subset_df = temp_df[temp_df['Indicator Name'] == indicator]
    # Transposing the df again to makes years the index.
    subset_df = subset_df.T

    # Plotting using the subset df.
    # in the same figure.
    plt.plot(subset_df[1:], label=country)

    # Labelling
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Saving the figure.
    #plt.savefig('figures/line_graph_'+indicator+'.png',
    #            bbox_inches='tight',
    #            dpi=200)
    plt.show()

    return subset_df


def main():
    
    # Create a list of Countries interested in evaluating.
    countries = ['China', 'India', 'Japan',
                 'United Kingdom', 'United States', 'Germany']
    
    wb_data_years, wb_data_country = load_data(
        'climate_change.csv',
        countries)

    subset_df,temp2, result = subset_data(wb_data_years,
                'CO2 emissions from solid fuel consumption (% of total)',
                'Agriculture, forestry, and fishing, value added (% of GDP)',
                'Urban population (% of total population)',
                'Cereal yield (kg per hectare)',
                countries)  
    
    # This produces a heatmap using cluster_tools py file.
    # The generated plot had to be saved maunally from spyder IDE.
    clst.map_corr(result)
    
    # Choosing indicators based on the heatmap
    indicator_1 = 'Urban population (% of total population)'
    indicator_2 = 'Cereal yield (kg per hectare)'
    
    cluster_df = result[[indicator_1, indicator_2]]
        
    scaler_output = clst.scaler(cluster_df)
    
    print(scaler_output[0])
    
    normalised_df = scaler_output[0]
    
    min_values = scaler_output[1]
    max_values = scaler_output[2]
   
    sil_score(normalised_df)
    
    cen, labels, cm = kmeans_cluster(6, normalised_df,
                   indicator_1,
                   indicator_2)

    backscaled_centers = clst.backscale(cen, min_values, max_values)
    x = backscaled_centers[:,0]
    y = backscaled_centers[:,1]
    
    plt.figure()
    plt.scatter(result[indicator_1], result[indicator_2],
                c=labels,cmap=cm )
    plt.scatter(x, y, marker="d", label="original centres", c='black' )
    
    plt.legend()
    

    indicator_df = generate_line_plot(wb_data_country ,
                       'India',
                       'Urban population (% of total population)',
                       'Years','urbanpop','Trialdank')
    indicator_df = indicator_df.reset_index()
    indicator_df = indicator_df.drop(index=0)
    
    indicator_df = indicator_df.rename(
        columns={'index': 'Year', 
                 'India': 'Urban population (% of total population)'})
    
    xdata = indicator_df['Year'].astype(int)
    ydata = indicator_df['Urban population (% of total population)'].astype(float)
    
    param, pcovar = opt.curve_fit(exp_growth, xdata, ydata, p0=(0,0))
    
    print(param, pcovar)
    
    indicator_df['pop_exp'] = exp_growth(xdata, *param)

    plt.figure()
    plt.plot(indicator_df['Year'],
             indicator_df['Urban population (% of total population)'], 
             label='Data')
    
    plt.plot(indicator_df['Year'], 
             indicator_df['pop_exp'], 
             label='fit')
    plt.xticks(rotation=90)

    plt.legend()
    plt.title('Trial 1')
    plt.show()

        
    return wb_data_years, wb_data_country, subset_df, temp2, result, indicator_df



if __name__ == '__main__':
    years, countries, susbset, fortemp, result, indicatordf = main()
