
import plotly.graph_objects as go

# Data for ROC curve
fpr = [0.0, 0.0, 0.0, 0.013888888888888888, 0.013888888888888888, 0.1388888888888889, 0.1388888888888889, 1.0]
tpr = [0.0, 0.023809523809523808, 0.9285714285714286, 0.9285714285714286, 0.9761904761904762, 0.9761904761904762, 1.0, 1.0]

# Create figure
fig = go.Figure()

# Add ROC curve
fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    mode='lines',
    name='ROC Curve',
    line=dict(color='#1FB8CD', width=3)
))

# Add diagonal reference line (random classifier)
fig.add_trace(go.Scatter(
    x=[0, 1],
    y=[0, 1],
    mode='lines',
    name='Random',
    line=dict(color='#DB4545', width=2, dash='dash')
))

# Update layout
fig.update_layout(
    title='ROC Curve',
    xaxis_title='FPR',
    yaxis_title='TPR',
    legend=dict(orientation='h', yanchor='bottom', y=1.05, xanchor='center', x=0.5)
)

# Update axes
fig.update_xaxes(range=[0, 1])
fig.update_yaxes(range=[0, 1])

# Update traces
fig.update_traces(cliponaxis=False)

# Save as PNG and SVG
fig.write_image('roc_curve.png')
fig.write_image('roc_curve.svg', format='svg')
