import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
import matplotlib.pyplot as plt

def scrape_bball_reference_splits(url):
    """
    Scrapes the splits table from a Basketball-Reference splits page.
    The splits table might be directly available or within commented HTML.
    Returns a tuple: (header, rows) from the table.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    table = soup.find('table', id='splits')
    
    if table is None:
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        splits_html = None
        for comment in comments:
            if 'table' in comment and 'id="splits"' in comment:
                splits_html = comment
                break
        if splits_html:
            soup_splits = BeautifulSoup(splits_html, 'html.parser')
            table = soup_splits.find('table', id='splits')
    
    if table is None:
        print("Splits table not found at URL:", url)
        return None, None

    header_tags = table.find('thead').find_all('tr')[-1].find_all('th')
    header = [th.get_text(strip=True) for th in header_tags]
    
    rows = []
    for row in table.find('tbody').find_all('tr'):
        if row.get('class') and 'thead' in row.get('class'):
            continue
        cells = row.find_all(['th', 'td'])
        row_data = [cell.get_text(separator="\n", strip=True) for cell in cells]
        rows.append(row_data)
    
    return header, rows

def get_overall_splits(url):
    """
    For a given splits URL, returns the "Overall" row as a DataFrame.
    If "Overall" is not found, defaults to the first row.
    """
    header, rows = scrape_bball_reference_splits(url)
    if header is None or rows is None:
        return None
    df = pd.DataFrame(rows, columns=header)
    if 'Split' not in df.columns:
        df.rename(columns={df.columns[0]: 'Split'}, inplace=True)
    overall = df[df['Split'] == 'Overall']
    if overall.empty:
        overall = df.iloc[[0]]  # Default to the first row if "Overall" is not found
    return overall

def extract_numeric(cell):
    """
    Extracts a single numeric value from a cell.
    If the cell contains multiple lines (e.g. "MP\n1825.0\nMP\n37.2"),
    this function returns the last number (assumed to be the per-game stat).
    """
    try:
        return float(cell)
    except (ValueError, TypeError):
        pass

    if isinstance(cell, str):
        parts = cell.split("\n")
        for part in reversed(parts):
            try:
                return float(part)
            except ValueError:
                continue
    if isinstance(cell, pd.Series):
        return extract_numeric(cell.iloc[-1])
    return np.nan

# Basketball-Reference splits URL for 2025 and their yearly salary data (in millions)
players_info = {
    'Devin Booker': {
        'url': "https://www.basketball-reference.com/players/b/bookede01/splits/",
        'Salary': 50.5
    },
    'Kevin Durant': {
        'url': "https://www.basketball-reference.com/players/d/duranke01/splits/",
        'Salary': 51.1
    },
    'Bradley Beal': {
        'url': "https://www.basketball-reference.com/players/b/bealbr01/splits/",
        'Salary': 50.2
    },
    'Jrue Holiday': {
        'url': "https://www.basketball-reference.com/players/h/holidjr01/splits/",
        'Salary': 30.0
    },
    'Jaylen Brown': {
        'url': "https://www.basketball-reference.com/players/b/brownja02/splits/",
        'Salary': 49.2
    },
    'Donnovan Mitchell': {
        'url': "https://www.basketball-reference.com/players/m/mitchdo01/splits/",
        'Salary': 34.8
    },
    'Dwanye Wade': {
        'url': "https://www.basketball-reference.com/players/w/wadedw01/splits/",
        'Salary': 30.1  # adjusted for inflation from 2016-17 season
    }
}

# For each player, scrape the Overall splits row and extract key performance metrics.
records = []
for player, info in players_info.items():
    url = info['url']
    salary = info['Salary']
    overall_df = get_overall_splits(url)
    if overall_df is None:
        print(f"Skipping {player} due to missing splits data.")
        continue

    row = overall_df.iloc[0]
    
    # Extract per-game stats using extract_numeric
    mp = extract_numeric(row.get('MP'))
    fg_pct = extract_numeric(row.get('FG%'))
    threep_pct = extract_numeric(row.get('3P%'))
    ft_pct = extract_numeric(row.get('FT%'))
    pts = extract_numeric(row.get('PTS'))
    TS = extract_numeric(row.get('TS%'))
    ORTG = extract_numeric(row.get('ORtg'))
    DRtG = extract_numeric(row.get('DRtg'))
    USG = extract_numeric(row.get('USG%'))
    
    records.append({
        'Player': player,
        'Salary': salary,
        'MP': mp,
        'FG%': fg_pct,
        '3P%': threep_pct,
        'FT%': ft_pct,
        'PTS': pts,
        'TS%': TS,
        'ORtg': ORTG,
        'DRtg': DRtG,
        'USG%': USG
    })

# Create a DataFrame for all players.
data = pd.DataFrame(records)
print("Aggregated Overall Splits Data:")
print(data)

# Regression: Does player performance correlate with salary?
features = ['MP', 'FG%', '3P%', 'FT%', 'PTS', 'TS%', 'USG%', 'ORtg', 'DRtg']
X = data[features]
y = data['Salary']

# Split data into training and testing sets (e.g., 70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ----- Standard Linear Regression -----
lr = LinearRegression()
lr.fit(X_train, y_train)
print("\nStandard Linear Regression:")
print("Train R²:", lr.score(X_train, y_train))
print("Test R²:", lr.score(X_test, y_test))
lr_cv_scores = cross_val_score(lr, X, y, cv=3)
print("Linear Regression CV scores:", lr_cv_scores)

# ----- Ridge Regression -----
ridge = Ridge(alpha=1.0)  # alpha can be tuned via cross-validation
ridge.fit(X_train, y_train)
print("\nRidge Regression:")
print("Train R²:", ridge.score(X_train, y_train))
print("Test R²:", ridge.score(X_test, y_test))
ridge_cv_scores = cross_val_score(ridge, X, y, cv=3)
print("Ridge Regression CV scores:", ridge_cv_scores)

# ----- Lasso Regression -----
lasso = Lasso(alpha=0.1)  # alpha can be tuned; too high may shrink coefficients to zero
lasso.fit(X_train, y_train)
print("\nLasso Regression:")
print("Train R²:", lasso.score(X_train, y_train))
print("Test R²:", lasso.score(X_test, y_test))
lasso_cv_scores = cross_val_score(lasso, X, y, cv=3)
print("Lasso Regression CV scores:", lasso_cv_scores)

# Create a composite performance index using all the performance metrics.
performance_cols = ['MP', 'FG%', '3P%', 'FT%', 'PTS', 'TS%', 'USG%', 'ORtg', 'DRtg']
scaler = MinMaxScaler()
scaled_performance = scaler.fit_transform(data[performance_cols])
data['Performance_Index'] = scaled_performance.mean(axis=1)

# Display the DataFrame with the composite performance index.
print("\nData with Performance Index:")
print(data[['Player', 'Performance_Index', 'Salary']])

# Plot Performance Index vs. Salary
plt.figure(figsize=(10, 6))
plt.scatter(data['Performance_Index'], data['Salary'], color='purple', s=100)
for i, player in enumerate(data['Player']):
    plt.annotate(player, 
                 (data['Performance_Index'].iloc[i], data['Salary'].iloc[i]),
                 textcoords="offset points", xytext=(5,5))
plt.xlabel("Composite Performance Index (Normalized)")
plt.ylabel("Salary (millions)")
plt.title("Player Performance vs. Salary")
plt.grid(True)
plt.show()
