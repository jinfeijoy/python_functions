from textwrap import wrap
import plotly.express as px
import plotly.graph_objects as go

def viz_scatter(number_of_graphs: int,
                df,
                fill: str,
                hovertemplate_name: list,
                color_list: list,
                title_one: str,
                title_two: str,
                y_title: str):
    '''

    The function accepts the following arguments:

    number_of_graphs - the number of objects that need to be placed on the chart (for example, two lines, etc.)
    df - dataframe
    fill - parameter to fill the area under the line
    hovertemplate_name - captions on hover
    color_list - line colors
    title_one - first chart name
    title_two - the second name of the chart
    y_title - y-axis label
    legend_x - x-axis position of the legend

    '''
    # Create an empty list to add data for graphs
    data = []

    # At each iteration, we create parameters for the chart objects
    for graph_number in range(number_of_graphs):
        graph = go.Scatter(
            hoverinfo='skip',  # Removing signatures when selecting
            x=df.iloc[:, 0],  # Passing data for the x-axis
            y=df.iloc[:, graph_number + 1],  # Passing data for the y-axis
            fill=fill,
            # Set your own format for pies on hover
            hovertemplate='<b>%{x}</b><br>' + f'<b>{hovertemplate_name[graph_number]}: </b>' + '%{y}<extra></extra>',
            # Setting the color
            marker_color=color_list[graph_number],
            # Setting a caption on hover
            name=hovertemplate_name[graph_number]
        )

        # Adding parameters to the list
        data.append(graph)

    # We transfer data for visualization
    fig = go.Figure(data)

    # Updating the chart settings when displaying
    fig.update_layout(
        title=f'<b>{"<br>".join(wrap(title_one, 70))}</b><br><sub>{title_two}</sub>',  # Passing the name of the chart
        xaxis_title='',  # Set the name of the x-axis
        yaxis_title=y_title,  # Set the name of the y-axis
        plot_bgcolor='rgba(0,0,0,0)',  # Setting the background color
        hovermode='x',  # Using the x-axis values for the records
        # Setting the legend parameters
        legend_orientation='h',
        # Setting parameters for the text
        font=dict(
            family='Arials',
            size=13,
            color='black'
        )
    )

    # Displaying the graph
    fig.show()