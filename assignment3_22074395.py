import cluster_tools as clst
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as skmet
import numpy as np
import scipy.optimize as opt
import errors as err

def load_data(dataset, country_list):
    """
    This function takes in a csv file and a list of countries that are of
    interest. Returns two dataframes one with the years as columns and the
    other country names as columns. (Reused assignment_2's code to
                                     load dataset')

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
        
        
        
    
    return result_df


def sil_score(normalised_df):
    
    for n in range(2,8):    

        ncluster = n
        
        kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
        
        kmeans.fit(normalised_df)
        
        labels = kmeans.labels_
                
        print("Nclusters: "+str(n))
        print(skmet.silhouette_score(normalised_df, labels))
        
def generate_kmeans_cluster_plot(result,normalised_df,cluster_df, indicator_1, 
                                 indicator_2, ncluster,
                                 min_values, max_values):
    
    
    kmeans = cluster.KMeans(n_clusters=ncluster, n_init=20)
    
    kmeans.fit(normalised_df)
    
    labels = kmeans.labels_
        
    cen = kmeans.cluster_centers_
    
    xkmeans = cen[:,0]
    ykmeans = cen[:,1]
    
    result['label'] = labels

        
    cm = plt.colormaps["Paired"]

    
    #cen, labels, cm = kmeans_cluster(ncluster, normalised_df,
    #               indicator_1,
    #               indicator_2)
    
    #xkmeans = cen[:,0]
    #ykmeans = cen[:,1]
    
    plt.figure()
    
    plt.scatter(normalised_df[indicator_1], 
                normalised_df[indicator_2], marker="o", c=labels,
                cmap=cm)
    plt.scatter(xkmeans, ykmeans, marker="d", label="kmeans centres", c='red' )

    plt.legend()
    plt.show()

    #cluster_df['labels'] = labels

    plt.figure()
    
    backscaled_centers = clst.backscale(cen, min_values, max_values)
    x = backscaled_centers[:,0]
    y = backscaled_centers[:,1]
    
    #plt.figure()
    
    
    cluster_name = []
    for i in range(ncluster):
        cluster_name.append(result[result['label'] == i]['Country'].unique())
      
    country_list = []    
    country_name = ""
    for i in range(len(cluster_name)):
        for j in range(len(cluster_name[i])):
            country_name += cluster_name[i][j] + ", "
        country_list.append(country_name)
        country_name = ""
        
    # Printing what countries Kmeans has clustered together   
    for i in range(len(country_list)):
        print("Kmeans Label "+str(i)+": "+ country_list[i])
            
    for i in result['label'].unique():
        plt.scatter(result[result['label'] == i][indicator_1],
                    result[result['label'] == i][indicator_2],
                    label=country_list[i])
        
    plt.scatter(x, y, marker="d", label="original centres", c='black' )
    
    plt.xlabel(indicator_1)
    plt.ylabel(indicator_2)    
    plt.legend()
    plt.show()
    
    
    
    return cluster_df
    

def exp_growth(t, scale, growth):
    """ Computes exponential function with scale and growth as free parameters
    """
    
    f = scale * np.exp(growth * (t)) 
    
    return f


def curve_fit_plot(data, country, indicator, xlabel, ylabel, title):
    """
    

    Returns None.

    """
    # Specifying figure size, as plot is big
    #plt.figure(figsize=(15, 8))

    temp_df = data[country].T
    # Subsetting the transposed df. Which now has years as columns
    subset_df = temp_df[temp_df['Indicator Name'] == indicator]
    # Transposing the df again to makes years the index.
    subset_df = subset_df.T

    subset_df = subset_df.reset_index()
    subset_df = subset_df.drop(index=0)
    
    subset_df = subset_df.rename(
        columns={'index': 'Year', 
                 country: indicator})
    
    
    xdata = subset_df['Year'].astype(int)
    ydata = subset_df[indicator].astype(float)
    
    param, pcovar = opt.curve_fit(exp_growth, xdata, ydata, p0=(0,0))
        
    subset_df['pop_exp'] = exp_growth(xdata, *param)
    pop_exp = exp_growth(xdata, *param)

    plt.figure()
    plt.plot(subset_df['Year'],
             subset_df[indicator], 
             label='Data')
    
    plt.plot(subset_df['Year'], 
             subset_df['pop_exp'], 
             label='fit')
    
    plt.xticks(rotation=90)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.legend()
    plt.title('Trial 1')
    plt.show()
    
    sigma = err.error_prop(xdata, exp_growth, param, pcovar)

    low = pop_exp - sigma
    up = pop_exp + sigma


    plt.figure()
    plt.title("exponenetial function error")
    plt.plot(xdata, ydata, label="data")
    plt.plot(xdata, pop_exp, label="fit")
    # plot error ranges with transparency
    plt.fill_between(xdata, low, up, alpha=0.5, color="red")

    plt.legend(loc="upper left")
    plt.show()


    return 


def main():
    
    # Create a list of Countries interested in evaluating.
    countries = ['China', 'India', 'Japan',
                 'United Kingdom', 'United States', 'Germany']
    
    wb_data_years, wb_data_country = load_data(
        'climate_change.csv',
        countries)

    result = subset_data(wb_data_years,
                'CO2 emissions from solid fuel consumption (% of total)',
                'Agriculture, forestry, and fishing, value added (% of GDP)',
                'Urban population (% of total population)',
                'Cereal yield (kg per hectare)',
                countries)  
    
    result = result.dropna(axis=0)
    
    # This produces a heatmap using cluster_tools py file.
    # The generated plot had to be saved maunally from spyder IDE.
    clst.map_corr(result)
    
    # Choosing indicators based on the heatmap
    indicator_2 = 'CO2 emissions from solid fuel consumption (% of total)'
    indicator_1 = 'Urban population (% of total population)'
    
    cluster_df = result[[indicator_1, indicator_2]]
    #cluster_df = cluster_df.dropna(axis=0)
    
    scaler_output = clst.scaler(cluster_df)
    
    print(scaler_output[0])
    
    normalised_df = scaler_output[0]
    
    min_values = scaler_output[1]
    max_values = scaler_output[2]
   
    sil_score(normalised_df)
    n_cluster = 2
    labeled_df = generate_kmeans_cluster_plot(result,normalised_df,cluster_df,
                                              indicator_1, indicator_2, 
                                 n_cluster, min_values, max_values)
    
    
    #print(result[result['label'] == 1]['Country'].unique())

    curve_fit_plot(wb_data_country,
                       'India',
                       'Urban population (% of total population)',
                       'Years','urbanpop','Trialdank')
        
    return wb_data_years, wb_data_country, result, labeled_df



if __name__ == '__main__':
    years, countries, result, labeled = main()
