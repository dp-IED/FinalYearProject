#!/usr/bin/env python3
"""
Generate HTML chart from comparison JSON results.
"""

import json
import sys
from pathlib import Path

def generate_html_chart(json_path: Path, output_path: Path):
    """Generate HTML chart from comparison JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    rows = data['rows']
    if len(rows) < 2:
        print(f"Error: Need at least 2 models to compare, found {len(rows)}")
        return 1
    
    # Extract model names
    model1 = rows[0]
    model2 = rows[1]
    model1_name = model1['model_id']
    model2_name = model2['model_id']
    
    # Key metrics to visualize
    window_metrics = {
        'window_accuracy': 'Window Accuracy',
        'window_f1': 'Window F1 Score',
        'window_f1_weighted': 'Window F1 (Weighted)',
        'window_f1_macro': 'Window F1 (Macro)',
        'window_precision': 'Window Precision',
        'window_precision_weighted': 'Window Precision (Weighted)',
        'window_recall': 'Window Recall',
        'window_recall_weighted': 'Window Recall (Weighted)',
    }
    
    sensor_metrics = {
        'sensor_accuracy': 'Sensor Accuracy',
        'sensor_f1': 'Sensor F1 Score',
        'sensor_precision': 'Sensor Precision',
        'sensor_recall': 'Sensor Recall',
    }
    
    efficiency_metrics = {
        'efficiency_llm_processing_time_seconds': 'LLM Processing Time (s)',
        'efficiency_total_processing_time_seconds': 'Total Processing Time (s)',
        'efficiency_windows_per_second': 'Windows per Second',
    }
    
    # Prepare data for charts
    def get_value(row, key):
        return row.get(key, 0) or 0
    
    window_data = {
        'labels': [window_metrics[k] for k in window_metrics.keys()],
        'model1': [get_value(model1, k) for k in window_metrics.keys()],
        'model2': [get_value(model2, k) for k in window_metrics.keys()],
    }
    
    sensor_data = {
        'labels': [sensor_metrics[k] for k in sensor_metrics.keys()],
        'model1': [get_value(model1, k) for k in sensor_metrics.keys()],
        'model2': [get_value(model2, k) for k in sensor_metrics.keys()],
    }
    
    efficiency_data = {
        'labels': [efficiency_metrics[k] for k in efficiency_metrics.keys()],
        'model1': [get_value(model1, k) for k in efficiency_metrics.keys()],
        'model2': [get_value(model2, k) for k in efficiency_metrics.keys()],
    }
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison: {model1_name} vs {model2_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            padding: 40px;
        }}
        h1 {{
            color: #333;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }}
        .subtitle {{
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }}
        .chart-container {{
            margin: 40px 0;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .chart-title {{
            font-size: 1.5em;
            color: #333;
            margin-bottom: 20px;
            font-weight: 600;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        canvas {{
            max-height: 400px;
        }}
        .legend {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .legend-label {{
            font-weight: 500;
            color: #555;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 40px 0;
        }}
        .summary-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .summary-card h3 {{
            font-size: 0.9em;
            opacity: 0.9;
            margin-bottom: 10px;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
        }}
        .summary-card .label {{
            font-size: 0.8em;
            opacity: 0.8;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Comparison Report</h1>
        <div class="subtitle">{model1_name} vs {model2_name}</div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Best Window F1</h3>
                <div class="value">{max(get_value(model1, 'window_f1'), get_value(model2, 'window_f1')):.4f}</div>
                <div class="label">{model1_name if get_value(model1, 'window_f1') > get_value(model2, 'window_f1') else model2_name}</div>
            </div>
            <div class="summary-card">
                <h3>Best Sensor F1</h3>
                <div class="value">{max(get_value(model1, 'sensor_f1'), get_value(model2, 'sensor_f1')):.4f}</div>
                <div class="label">{model1_name if get_value(model1, 'sensor_f1') > get_value(model2, 'sensor_f1') else model2_name}</div>
            </div>
            <div class="summary-card">
                <h3>Fastest Processing</h3>
                <div class="value">{min(get_value(model1, 'efficiency_total_processing_time_seconds'), get_value(model2, 'efficiency_total_processing_time_seconds')):.2f}s</div>
                <div class="label">{model1_name if get_value(model1, 'efficiency_total_processing_time_seconds') < get_value(model2, 'efficiency_total_processing_time_seconds') else model2_name}</div>
            </div>
            <div class="summary-card">
                <h3>Highest Throughput</h3>
                <div class="value">{max(get_value(model1, 'efficiency_windows_per_second'), get_value(model2, 'efficiency_windows_per_second')):.2f}</div>
                <div class="label">{model1_name if get_value(model1, 'efficiency_windows_per_second') > get_value(model2, 'efficiency_windows_per_second') else model2_name}</div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Window-Level Metrics</div>
            <canvas id="windowChart"></canvas>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(54, 162, 235, 0.8);"></div>
                    <span class="legend-label">{model1_name}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(255, 99, 132, 0.8);"></div>
                    <span class="legend-label">{model2_name}</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Sensor-Level Metrics</div>
            <canvas id="sensorChart"></canvas>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(54, 162, 235, 0.8);"></div>
                    <span class="legend-label">{model1_name}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(255, 99, 132, 0.8);"></div>
                    <span class="legend-label">{model2_name}</span>
                </div>
            </div>
        </div>
        
        <div class="chart-container">
            <div class="chart-title">Efficiency Metrics</div>
            <canvas id="efficiencyChart"></canvas>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(54, 162, 235, 0.8);"></div>
                    <span class="legend-label">{model1_name}</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background: rgba(255, 99, 132, 0.8);"></div>
                    <span class="legend-label">{model2_name}</span>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const windowData = {{
            labels: {json.dumps(window_data['labels'])},
            datasets: [
                {{
                    label: {json.dumps(model1_name)},
                    data: {json.dumps(window_data['model1'])},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }},
                {{
                    label: {json.dumps(model2_name)},
                    data: {json.dumps(window_data['model2'])},
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2
                }}
            ]
        }};
        
        const sensorData = {{
            labels: {json.dumps(sensor_data['labels'])},
            datasets: [
                {{
                    label: {json.dumps(model1_name)},
                    data: {json.dumps(sensor_data['model1'])},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }},
                {{
                    label: {json.dumps(model2_name)},
                    data: {json.dumps(sensor_data['model2'])},
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2
                }}
            ]
        }};
        
        const efficiencyData = {{
            labels: {json.dumps(efficiency_data['labels'])},
            datasets: [
                {{
                    label: {json.dumps(model1_name)},
                    data: {json.dumps(efficiency_data['model1'])},
                    backgroundColor: 'rgba(54, 162, 235, 0.8)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2
                }},
                {{
                    label: {json.dumps(model2_name)},
                    data: {json.dumps(efficiency_data['model2'])},
                    backgroundColor: 'rgba(255, 99, 132, 0.8)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2
                }}
            ]
        }};
        
        const chartOptions = {{
            responsive: true,
            maintainAspectRatio: true,
            plugins: {{
                legend: {{
                    display: false
                }},
                tooltip: {{
                    callbacks: {{
                        label: function(context) {{
                            return context.dataset.label + ': ' + context.parsed.y.toFixed(4);
                        }}
                    }}
                }}
            }},
            scales: {{
                y: {{
                    beginAtZero: true,
                    ticks: {{
                        precision: 4
                    }}
                }}
            }}
        }};
        
        new Chart(document.getElementById('windowChart'), {{
            type: 'bar',
            data: windowData,
            options: chartOptions
        }});
        
        new Chart(document.getElementById('sensorChart'), {{
            type: 'bar',
            data: sensorData,
            options: chartOptions
        }});
        
        new Chart(document.getElementById('efficiencyChart'), {{
            type: 'bar',
            data: efficiencyData,
            options: chartOptions
        }});
    </script>
</body>
</html>"""
    
    output_path.write_text(html, encoding='utf-8')
    print(f"✓ HTML chart saved to: {output_path}")
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: generate_comparison_html.py <compare.json> [output.html]")
        print("  If output.html is not specified, uses compare.html in the same directory")
        return 1
    
    json_path = Path(sys.argv[1])
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return 1
    
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = json_path.parent / "compare.html"
    
    return generate_html_chart(json_path, output_path)


if __name__ == "__main__":
    sys.exit(main())
