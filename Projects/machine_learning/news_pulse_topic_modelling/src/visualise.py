def generate_topic_barchart(topic_model, top_n=6):
    # I create a bar chart to display the highest-scoring words for my top topics.
    # I return the figure object so it can be rendered by my dashboard later.
    fig = topic_model.visualize_barchart(top_n_topics=top_n)
    return fig

def generate_distance_map(topic_model):
    # I generate a 2D map showing the semantic distance between all my discovered topics.
    # Topics that cluster closely together share similar themes.
    fig = topic_model.visualize_topics()
    return fig


def generate_drift_chart(topic_model, topics_over_time, selected_topics=None, top_n=5):
    # I check if the user has selected specific topics from the dashboard.
    # If they have, I only plot those exact topics to keep the chart clean.
    if selected_topics:
        fig = topic_model.visualize_topics_over_time(topics_over_time, topics=selected_topics)
    # If no topics are selected, I default to showing the top N topics.
    else:
        fig = topic_model.visualize_topics_over_time(topics_over_time, top_n_topics=top_n)

    return fig