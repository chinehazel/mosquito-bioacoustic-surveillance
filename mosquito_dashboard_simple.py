#!/usr/bin/env python3
"""
========================================
MOSQUITO DETECTION DASHBOARD - LIGHT MODE
========================================
"""

import re
import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from datetime import datetime
import json
from pathlib import Path
from collections import deque

LOG_DIR         = str(Path.home() / 'Library/CloudStorage/GoogleDrive-mosquitodetect@gmail.com/My Drive/Mosquito_detect_mac/logs')
UPDATE_INTERVAL = 2000
MAX_HISTORY     = 100
MAX_LOG_LINES   = 50

# Regex to parse live_log.txt lines
# e.g. [2026-06-03 21:36:23]  Unknown/Non-dengue Carrier Mosquito - saved as capture_0001.npy
_LOG_RE = re.compile(
    r'\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s+(.+?) - saved as'
)

# ========================================
# DATA MONITOR
# ========================================
class DataMonitor:
    def __init__(self):
        self.log_dir          = Path(LOG_DIR)
        self.data_buffer      = deque(maxlen=MAX_HISTORY)   # from .jsonl (Aedes only)
        self.live_buffer      = []                          # from live_log.txt (all detections)
        self.current_log_file = None
        self.last_position    = 0
        self.stats = {
            'total': 0, 'mosquito': 0, 'background': 0, 'species': {}
        }

    def find_latest_log(self):
        if not self.log_dir.exists():
            return None
        stats_file = self.log_dir / 'stats.json'
        if stats_file.exists():
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                updated_at = stats.get('updated_at', '')
                if updated_at:
                    session_date = updated_at[:10].replace('-', '')
                    log_files = list(self.log_dir.glob(f'detections_{session_date}_*.jsonl'))
                    if log_files:
                        return max(log_files, key=lambda x: x.stat().st_mtime)
                    return None
            except:
                pass
        log_files = list(self.log_dir.glob('detections_*.jsonl'))
        if not log_files:
            return None
        return max(log_files, key=lambda x: x.stat().st_mtime)

    def _read_stats_file(self):
        stats_file = self.log_dir / 'stats.json'
        if not stats_file.exists():
            return
        try:
            with open(stats_file, 'r') as f:
                data = json.load(f)
            self.stats['total']      = data.get('total', 0)
            self.stats['mosquito']   = data.get('mosquito', 0)
            self.stats['background'] = data.get('background', 0)
            self.stats['species']    = data.get('species', {})
        except:
            pass

    def _read_live_log(self):
        """Parse live_log.txt into structured detection entries (excludes Background)."""
        live_log_file = self.log_dir / 'live_log.txt'
        if not live_log_file.exists():
            self.live_buffer = []
            return
        try:
            with open(live_log_file, 'r', encoding='utf-8') as f:
                content = f.read()
            entries = []
            for match in _LOG_RE.finditer(content):
                ts    = match.group(1)
                label = match.group(2).strip()
                if label == 'Background Sound':
                    continue
                entries.append({
                    'timestamp': ts,
                    'classification': {
                        'final_classification': label,
                    }
                })
            self.live_buffer = entries
        except:
            self.live_buffer = []

    def update(self):
        self._read_stats_file()
        self._read_live_log()
        # Also read .jsonl for confirmed Aedes (confidence data)
        latest = self.find_latest_log()
        if latest != self.current_log_file:
            self.current_log_file = latest
            self.last_position    = 0
            self.data_buffer.clear()
        if not self.current_log_file:
            return
        try:
            with open(self.current_log_file, 'r') as f:
                f.seek(self.last_position)
                new_lines          = f.readlines()
                self.last_position = f.tell()
            for line in new_lines:
                if line.strip():
                    self.data_buffer.append(json.loads(line))
        except:
            pass

    def get_timeline_data(self, n=50):
        """
        For timeline: prefer .jsonl (has confidence) for Aedes detections,
        supplement with live_log entries for Unknown detections.
        Returns list of dicts with 'timestamp' and 'final_classification'.
        """
        # Merge: .jsonl entries (Aedes, full data) + live_buffer (all mosquitoes)
        # Since .jsonl only has Aedes and live_buffer has everything,
        # use live_buffer as the primary source for the timeline.
        return self.live_buffer[-n:]

    def get_recent_detections(self, n=10):
        """For the table: show most recent n mosquito detections from live_log."""
        return self.live_buffer[-n:]

    def get_stats(self):
        return self.stats

    def get_live_log(self, n=MAX_LOG_LINES):
        live_log_file = self.log_dir / 'live_log.txt'
        if not live_log_file.exists():
            return ["Waiting for system to start..."]
        try:
            with open(live_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            return [l.rstrip() for l in reversed(lines[-n:])]
        except:
            return ["Error reading live log"]


# ========================================
# DASH APP
# ========================================
app       = dash.Dash(__name__)
app.title = "MosquitoDetect"
monitor   = DataMonitor()

BG_PAGE   = '#faf8f5'
BG_CARD   = '#ffffff'
BG_CARD2  = '#fffbf5'
AMBER     = '#d97706'
ORANGE    = '#ea580c'
DANGER    = '#dc2626'
MUTED     = '#a89070'
TEXT_PRI  = '#1c1008'
TEXT_SEC  = '#6b4c1e'
BORDER    = '#e8ddd0'
FONT      = "'Public Sans', sans-serif"

CARD_STYLE = {
    'backgroundColor': BG_CARD,
    'border':          f'1px solid {BORDER}',
    'borderRadius':    '10px',
    'padding':         '20px 24px',
    'flex':            '1',
    'boxShadow':       '0 1px 4px rgba(0,0,0,0.06)',
}

CHART_LAYOUT_BASE = dict(
    paper_bgcolor = BG_CARD,
    plot_bgcolor  = BG_CARD,
    font          = dict(color=TEXT_PRI, family='Public Sans, sans-serif', size=11),
    margin        = dict(l=40, r=20, t=44, b=40),
    height        = 340,
    xaxis         = dict(gridcolor='#f0e8de', zerolinecolor='#f0e8de', color=TEXT_SEC),
    yaxis         = dict(gridcolor='#f0e8de', zerolinecolor='#f0e8de', color=TEXT_SEC),
)

app.layout = html.Div([

    html.Div([
        html.Div([
            html.Span("MosquitoDetect", style={
                'fontFamily': FONT, 'fontWeight': '700',
                'fontSize': '17px', 'color': AMBER,
            }),
            html.P(
                "IoT Bioacoustic Surveillance  \u00b7  Dengue Vector Monitoring  \u00b7",
                style={
                    'margin': '3px 0 0 0', 'fontSize': '11px',
                    'color': MUTED, 'fontFamily': FONT,
                }
            ),
        ], style={'flex': '1'}),

        html.Div([
            html.Span("\u25cf", style={
                'color': '#ef4444', 'fontSize': '13px',
                'marginRight': '6px', 'animation': 'pulse 1.5s infinite',
            }),
            html.Span("LIVE", style={
                'fontFamily': FONT, 'fontWeight': '700',
                'fontSize': '13px', 'color': '#ef4444', 'letterSpacing': '2px',
            }),
        ], style={
            'display':         'flex',
            'alignItems':      'center',
            'backgroundColor': '#fff1f1',
            'border':          '1px solid #fca5a5',
            'borderRadius':    '6px',
            'padding':         '5px 12px',
        }),

    ], style={
        'display': 'flex', 'alignItems': 'center',
        'justifyContent': 'space-between',
        'backgroundColor': BG_CARD,
        'borderBottom': f'2px solid {AMBER}',
        'padding': '14px 28px',
        'boxShadow': '0 1px 6px rgba(0,0,0,0.07)',
    }),

    html.Div([

        html.Div([
            html.Div("TOTAL TRANSMISSIONS", style={
                'fontFamily': FONT, 'fontWeight': '600',
                'fontSize': '10px', 'letterSpacing': '1.2px',
                'color': MUTED, 'marginBottom': '8px',
            }),
            html.Div(id='total-count', children='\u2014', style={
                'fontSize': '40px', 'fontWeight': '700',
                'color': TEXT_PRI, 'lineHeight': '1', 'fontFamily': FONT,
            }),
            html.Div("transmissions logged", style={
                'fontSize': '11px', 'color': MUTED,
                'fontFamily': FONT, 'marginTop': '6px',
            }),
        ], style={**CARD_STYLE}),

        html.Div([
            html.Div("MOSQUITOES DETECTED", style={
                'fontFamily': FONT, 'fontWeight': '600',
                'fontSize': '10px', 'letterSpacing': '1.2px',
                'color': MUTED, 'marginBottom': '8px',
            }),
            html.Div(id='mosquito-count', children='\u2014', style={
                'fontSize': '40px', 'fontWeight': '700',
                'color': DANGER, 'lineHeight': '1', 'fontFamily': FONT,
            }),
            html.Div("positive detections", style={
                'fontSize': '11px', 'color': MUTED,
                'fontFamily': FONT, 'marginTop': '6px',
            }),
        ], style={**CARD_STYLE}),

        html.Div([
            html.Div("BACKGROUND", style={
                'fontFamily': FONT, 'fontWeight': '600',
                'fontSize': '10px', 'letterSpacing': '1.2px',
                'color': MUTED, 'marginBottom': '8px',
            }),
            html.Div(id='background-count', children='\u2014', style={
                'fontSize': '40px', 'fontWeight': '700',
                'color': TEXT_SEC, 'lineHeight': '1', 'fontFamily': FONT,
            }),
            html.Div("non-target audio", style={
                'fontSize': '11px', 'color': MUTED,
                'fontFamily': FONT, 'marginTop': '6px',
            }),
        ], style={**CARD_STYLE}),

        html.Div([
            html.Div("DETECTION RATE", style={
                'fontFamily': FONT, 'fontWeight': '600',
                'fontSize': '10px', 'letterSpacing': '1.2px',
                'color': MUTED, 'marginBottom': '8px',
            }),
            html.Div(id='detection-rate', children='\u2014', style={
                'fontSize': '40px', 'fontWeight': '700',
                'color': AMBER, 'lineHeight': '1', 'fontFamily': FONT,
            }),
            html.Div("positive identification", style={
                'fontSize': '11px', 'color': MUTED,
                'fontFamily': FONT, 'marginTop': '6px',
            }),
        ], style={**CARD_STYLE}),

    ], style={
        'display': 'flex', 'gap': '12px',
        'padding': '16px 20px', 'backgroundColor': BG_PAGE,
    }),

    html.Div([
        html.Div([dcc.Graph(id='timeline-chart', config={'displayModeBar': False})],
                 style={'flex': '1', 'backgroundColor': BG_CARD,
                        'border': f'1px solid {BORDER}', 'borderRadius': '10px',
                        'overflow': 'hidden', 'boxShadow': '0 1px 4px rgba(0,0,0,0.05)'}),
        html.Div([dcc.Graph(id='species-pie', config={'displayModeBar': False})],
                 style={'flex': '1', 'backgroundColor': BG_CARD,
                        'border': f'1px solid {BORDER}', 'borderRadius': '10px',
                        'overflow': 'hidden', 'boxShadow': '0 1px 4px rgba(0,0,0,0.05)'}),
    ], style={'display': 'flex', 'gap': '12px', 'padding': '0 20px 12px', 'backgroundColor': BG_PAGE}),

    html.Div([
        html.Div([dcc.Graph(id='recent-table', config={'displayModeBar': False})],
                 style={'flex': '1', 'backgroundColor': BG_CARD,
                        'border': f'1px solid {BORDER}', 'borderRadius': '10px',
                        'overflow': 'hidden', 'boxShadow': '0 1px 4px rgba(0,0,0,0.05)'}),
    ], style={'display': 'flex', 'gap': '12px', 'padding': '0 20px 12px', 'backgroundColor': BG_PAGE}),

    html.Div([
        html.Div([
            html.Span("Live Detection Log", style={
                'fontFamily': FONT, 'fontWeight': '600',
                'fontSize': '12px', 'color': AMBER,
            }),
            html.Span("  \u2014  real-time system output", style={
                'fontFamily': FONT, 'fontSize': '11px', 'color': MUTED,
            }),
        ], style={'marginBottom': '10px'}),

        html.Pre(
            id='live-log',
            children='Waiting for detections...',
            style={
                'backgroundColor': '#111111',
                'color':           '#f0f0f0',
                'fontFamily':      "'Courier New', monospace",
                'fontSize':        '12px',
                'padding':         '14px 16px',
                'borderRadius':    '8px',
                'border':          '1px solid #333333',
                'height':          '180px',
                'overflowY':       'auto',
                'whiteSpace':      'pre-wrap',
                'margin':          0,
                'lineHeight':      '1.6',
            }
        )
    ], style={
        'backgroundColor': BG_CARD,
        'border':          f'1px solid {BORDER}',
        'borderRadius':    '10px',
        'padding':         '16px 20px',
        'margin':          '0 20px 20px',
        'boxShadow':       '0 1px 4px rgba(0,0,0,0.05)',
    }),

    dcc.Interval(id='interval', interval=UPDATE_INTERVAL, n_intervals=0),

], style={
    'fontFamily': FONT, 'backgroundColor': BG_PAGE, 'minHeight': '100vh',
})

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
        {%css%}
        <style>
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50%       { opacity: 0.25; }
            }
            body { margin: 0; font-family: 'Public Sans', sans-serif; }
            ::-webkit-scrollbar { width: 5px; height: 5px; }
            ::-webkit-scrollbar-track { background: #faf8f5; }
            ::-webkit-scrollbar-thumb { background: #e0c9a6; border-radius: 3px; }
            ::-webkit-scrollbar-thumb:hover { background: #d97706; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''


# ========================================
# CALLBACKS
# ========================================

@app.callback(
    [Output('total-count',      'children'),
     Output('mosquito-count',   'children'),
     Output('background-count', 'children'),
     Output('detection-rate',   'children')],
    [Input('interval', 'n_intervals')]
)
def update_stats(n):
    monitor.update()
    stats      = monitor.get_stats()
    total      = stats['total']
    mosquito   = stats['mosquito']
    background = stats['background']
    rate       = f"{mosquito / max(1, total) * 100:.1f}%"
    return str(total), str(mosquito), str(background), rate


@app.callback(
    Output('timeline-chart', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_timeline(n):
    data   = monitor.get_timeline_data(50)
    layout = {**CHART_LAYOUT_BASE, 'title': dict(
        text="Detection Timeline",
        font=dict(color=TEXT_PRI, size=13, family='Public Sans, sans-serif'),
        x=0.03, xanchor='left'
    )}

    if not data:
        return go.Figure().update_layout(**layout,
            annotations=[dict(text="No mosquito detections yet", x=0.5, y=0.5,
                              xref='paper', yref='paper', showarrow=False,
                              font=dict(color=MUTED, size=13, family='Public Sans'))])

    species_colors = {
        'Aedes aegypti':                        '#dc2626',
        'Aedes albopictus':                     '#ea580c',
        'Anopheles gambiae':                    '#d97706',
        'Anopheles arabiensis':                 '#b45309',
        'Culex quinquefasciatus':               '#f59e0b',
        'Culex pipiens':                        '#fb923c',
        'Unknown/Non-dengue Carrier Mosquito':  '#a89070',
    }

    times  = [datetime.strptime(e['timestamp'], '%Y-%m-%d %H:%M:%S') for e in data]
    labels = [e['classification']['final_classification'] for e in data]
    colors = [species_colors.get(l, AMBER) for l in labels]

    fig = go.Figure(data=[go.Scatter(
        x=times, y=labels, mode='markers',
        marker=dict(size=11, color=colors,
                    line=dict(width=1, color='rgba(255,255,255,0.6)')),
        text=labels, hoverinfo='text+x'
    )])
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output('species-pie', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_pie(n):
    # Count from live_buffer (all mosquito detections)
    data = monitor.live_buffer
    layout = {**CHART_LAYOUT_BASE, 'title': dict(
        text="Species Distribution",
        font=dict(color=TEXT_PRI, size=13, family='Public Sans, sans-serif'),
        x=0.03, xanchor='left'
    )}

    if not data:
        return go.Figure().update_layout(**layout,
            annotations=[dict(text="No mosquito detections yet", x=0.5, y=0.5,
                              xref='paper', yref='paper', showarrow=False,
                              font=dict(color=MUTED, size=13, family='Public Sans'))])

    counts = {}
    for e in data:
        label = e['classification']['final_classification']
        counts[label] = counts.get(label, 0) + 1

    labels = list(counts.keys())
    values = list(counts.values())

    color_map = {
        'Aedes aegypti':                        '#dc2626',
        'Aedes albopictus':                     '#ea580c',
        'Anopheles gambiae':                    '#d97706',
        'Anopheles arabiensis':                 '#b45309',
        'Culex quinquefasciatus':               '#f59e0b',
        'Culex pipiens':                        '#fb923c',
        'Unknown/Non-dengue Carrier Mosquito':  '#a89070',
    }
    colors = [color_map.get(l, AMBER) for l in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels, values=values, hole=0.42,
        textinfo='label+percent',
        textfont=dict(family='Public Sans, sans-serif', size=11),
        marker=dict(colors=colors, line=dict(color='#ffffff', width=2))
    )])
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output('recent-table', 'figure'),
    [Input('interval', 'n_intervals')]
)
def update_table(n):
    data   = monitor.get_recent_detections(10)
    layout = {**CHART_LAYOUT_BASE, 'title': dict(
        text="Recent Detections",
        font=dict(color=TEXT_PRI, size=13, family='Public Sans, sans-serif'),
        x=0.03, xanchor='left'
    )}

    if not data:
        return go.Figure().update_layout(**layout,
            annotations=[dict(text="No mosquito detections yet", x=0.5, y=0.5,
                              xref='paper', yref='paper', showarrow=False,
                              font=dict(color=MUTED, size=13, family='Public Sans'))])

    times, classifications = [], []
    for entry in reversed(data):
        ts = entry['timestamp']
        try:
            times.append(datetime.fromisoformat(ts).strftime('%H:%M:%S'))
        except:
            times.append(datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').strftime('%H:%M:%S'))
        classifications.append(entry['classification']['final_classification'])

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>TIME</b>', '<b>CLASSIFICATION</b>'],
            fill_color=AMBER,
            font=dict(color='#ffffff', size=11, family='Public Sans, sans-serif'),
            align='center',
            line_color=BORDER,
            height=34,
        ),
        cells=dict(
            values=[times, classifications],
            fill_color=[[BG_CARD2 if i % 2 == 0 else BG_CARD for i in range(len(times))]],
            font=dict(color=TEXT_PRI, size=11, family='Public Sans, sans-serif'),
            align='center',
            line_color=BORDER,
            height=28,
        )
    )])
    fig.update_layout(**layout)
    return fig


@app.callback(
    Output('live-log', 'children'),
    [Input('interval', 'n_intervals')]
)
def update_live_log(n):
    return '\n'.join(monitor.get_live_log(MAX_LOG_LINES))


# ========================================
# MAIN
# ========================================
if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("MOSQUITO DETECTION DASHBOARD — LIGHT MODE")
    print("=" * 50)
    print("Starting at http://localhost:8050")
    print("Press Ctrl+C to stop")
    print("=" * 50 + "\n")
    app.run(debug=True, host='0.0.0.0', port=8050)