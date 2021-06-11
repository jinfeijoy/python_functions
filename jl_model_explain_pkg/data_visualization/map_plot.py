import plotly.express as px
import pandas as pd
def static_scatter_map(data, feature, country, continent, title, subtitle):
    """

    :param data: dataset
    :param feature: feature to be shown in the plot
    :param country: country variable name
    :param continent: continent variable name
    :param title: title
    :param subtitle: subtitle
    :return:
    """
    df = data[[country, feature, continent]]
    fig = px.scatter_geo(
         df, # Passing the dataframe
         locations=country, # Select the column with the name of the countries
         color=continent,
         locationmode='country names', # We pass the parameter of determining the country on the map (by name)
         hover_name=country,  # Passing values for the signature on hover
         size=feature # Passing a column with values
    )
    fig.update_layout(
    # Set the name of the map
    title_text=title+' <br><sub>'+subtitle+'</sub>',
    legend_orientation='h', # Place the legend caption under the chart
    legend_title_text='', # Remove the name of the legend group
    # Determine the map display settings (remove the frame, etc.)
    geo=dict(
       showframe=False,
       showcoastlines=False,
       projection_type='equirectangular'
    ),
    # Setting parameters for the text
    font=dict(
       family='TimesNewRoman',
       size=18,
       color='black'
    )
    )

    fig.show()

def static_choro_map(data, feature, country, title, subtitle, colors = 'reds'):
    """

    :param data: dataset
    :param feature: feature to be shown in the plot
    :param country: country variable name
    :param title: title
    :param subtitle: subtitle
    :param colors: continuous color settting: e.g. reds, blues
    :return:
    """
    df = data[[country, feature]]
    fig = px.choropleth(
         df, # Passing the dataframe
         locations=country, # Select the column with the name of the countries
         color=feature,
         locationmode='country names', # We pass the parameter of determining the country on the map (by name)
         hover_name=country,  # Passing values for the signature on hover
         color_continuous_scale=colors # Passing a column with values
    )
    fig.update_layout(
    # Set the name of the map
        title_text=title+' <br><sub>'+subtitle+'</sub>',
        geo=dict(
           showframe=False,
           showcoastlines=False,
           projection_type='equirectangular'
        ),
           # Setting parameters for the text
           font=dict(
           family='Arials',
           size=13,
           color='black'
        )
    )
    fig.show()

    def dynamic_choro_map(data, iso_code, country, feature, date, title):
        df = data[[iso_code, feature, date, country]]
        df = df.sort_values(date, ascending=True)
        df['date'] = pd.to_datetime(df[date], errors='coerce').dt.strftime('%m-%d-%Y')
        color_range = int(df[feature].quantile(0.95))
        fig = px.choropleth(
            df,  # Input Dataframe
            locations=iso_code,  # identify country code column
            color=feature,  # identify representing column
            hover_name=country,  # identify hover name
            animation_frame=date,
            color_continuous_scale='viridis',
            projection="natural earth",  # select projection
            range_color=[0, color_range],
            title='<span style="font-size:36px; font-family:Times New Roman">' + title,
        )  # select range of dataset
        fig.show()

def dynamic_choro_map(data, iso_code, country, feature, date, title):
    """

    :param data: dataset
    :param iso_code: country's iso code variable name
    :param country: country variable name
    :param feature: feature need to be shown in the plot
    :param date: date (object)
    :param title: title
    :return:
    """
    df = data[[iso_code, feature, date,country]]
    df = df.sort_values(date, ascending = True)
    df['date'] = pd.to_datetime(df[date], errors='coerce').dt.strftime('%m-%d-%Y')
    color_range = int(df[feature].quantile(0.95))
    fig = px.choropleth(
        df,                            # Input Dataframe
        locations=iso_code,           # identify country code column
        color=feature,                     # identify representing column
        hover_name=country,              # identify hover name
        animation_frame=date,
        color_continuous_scale= 'viridis',
        projection="natural earth",        # select projection
        range_color=[0,color_range],
        title='<span style="font-size:36px; font-family:Times New Roman">'+title,
    )             # select range of dataset
    fig.show()