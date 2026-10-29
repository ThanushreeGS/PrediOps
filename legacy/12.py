import IPython

# Read and render the generated HTML report
with open('/content/aiops_capacity/aiops_capacity/reports/operator_report.html', 'r') as f:
    html_content = f.read()

IPython.display.HTML(html_content)
