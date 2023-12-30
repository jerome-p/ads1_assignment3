import cluster_tools as clst
import pandas as pd
import sklearn.cluster as cluster
import matplotlib.pyplot as plt
import sklearn.metrics as skmet


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

def main():
    
    # Create a list of Countries interested in evaluating.
    countries = ['China', 'India', 'Japan',
                 'United Kingdom', 'United States', 'Germany']
    
    wb_data_years, wb_data_country = load_data(
        'agriculture_rural_dev.csv',
        countries)

    subset_df,temp2, result = subset_data(wb_data_years,
                'Rural population (% of total population)',
                'Surface area (sq. km)',
                'Cereal production (metric tons)',
                'Land area (sq. km)',
                countries)  
    
    clst.map_corr(result)

    
    temp_df = subset_df.copy()
    temp_df = temp_df.set_index('Indicator Name')
    
    temp_df = temp_df.T
        
    return wb_data_years, wb_data_country, subset_df, temp_df, temp2, result



if __name__ == '__main__':
    years, countries, susbset,temp, fortemp, result = main()
