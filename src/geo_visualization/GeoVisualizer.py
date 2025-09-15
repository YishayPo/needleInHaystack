import pandas as pd
import plotly.express as px
import os

# --- Data Dictionaries ---
STATE_CORPORATE_TAX_RATES = {
    'AL': 0.065, 'AK': 0.094, 'AZ': 0.049, 'AR': 0.048, 'CA': 0.0884, 'CO': 0.044,
    'CT': 0.075, 'DE': 0.087, 'FL': 0.055, 'GA': 0.0575, 'HI': 0.064, 'ID': 0.058,
    'IL': 0.095, 'IN': 0.049, 'IA': 0.055, 'KS': 0.07, 'KY': 0.05, 'LA': 0.075,
    'ME': 0.0893, 'MD': 0.0825, 'MA': 0.08, 'MI': 0.06, 'MN': 0.098, 'MS': 0.05,
    'MO': 0.04, 'MT': 0.0675, 'NE': 0.0584, 'NV': 0.0, 'NH': 0.075, 'NJ': 0.09,
    'NM': 0.059, 'NY': 0.0725, 'NC': 0.025, 'ND': 0.0431, 'OH': 0.0, 'OK': 0.04,
    'OR': 0.076, 'PA': 0.0849, 'RI': 0.07, 'SC': 0.05, 'SD': 0.0, 'TN': 0.065,
    'TX': 0.0, 'UT': 0.0465, 'VT': 0.085, 'VA': 0.06, 'WA': 0.0, 'WV': 0.065,
    'WI': 0.079, 'WY': 0.0
}

# Source: 2024 US Presidential Election Results. Margin = (Republican % - Democrat %)
STATE_POLITICAL_LEANING_2024 = {
    'AL': 27.8, 'AK': 12.1, 'AZ': 2.6, 'AR': 29.9, 'CA': -28.9, 'CO': -14.1, 'CT': -19.8,
    'DE': -18.5, 'FL': 7.8, 'GA': 2.4, 'HI': -29.1, 'ID': 33.1, 'IL': -16.5, 'IN': 18.2,
    'IA': 10.2, 'KS': 16.3, 'KY': 27.8, 'LA': 20.1, 'ME': -8.4, 'MD': -32.5, 'MA': -32.8,
    'MI': 0.3, 'MN': -5.8, 'MS': 18.2, 'MO': 17.3, 'MT': 18.1, 'NE': 20.5, 'NV': 2.1,
    'NH': -6.9, 'NJ': -15.1, 'NM': -9.8, 'NY': -22.5, 'NC': 3.6, 'ND': 35.1, 'OH': 10.4,
    'OK': 34.9, 'OR': -15.8, 'PA': 1.2, 'RI': -20.1, 'SC': 13.2, 'SD': 28.3, 'TN': 25.1,
    'TX': 9.1, 'UT': 22.3, 'VT': -34.9, 'VA': -9.5, 'WA': -18.7, 'WV': 40.1, 'WI': 0.7,
    'WY': 45.8
}

STATE_NAMES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
    'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland', 'MA': 'Massachusetts',
    'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi', 'MO': 'Missouri', 'MT': 'Montana',
    'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico',
    'NY': 'New York', 'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
    'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
    'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont',
    'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming'
}


class GeoVisualizer:
    """
    A class to create geographic and correlation visualizations of company market data by state.
    It encapsulates data loading, cleaning, and calculation logic.
    """

    def __init__(self, file_path: str):
        """
        Initializes the GeoVisualizer with the path to the company data CSV file.
        """
        self.file_path = file_path
        self.dataframe = None
        print(f"GeoVisualizer initialized for file: {self.file_path}")

    def load_and_prepare_data(self) -> bool:
        """
        Loads and prepares the company data from the CSV file.
        """
        try:
            print("Attempting to load and prepare data...")
            df = pd.read_csv(self.file_path)
            df_usa = df[df['hq_country'] == 'United States'].copy()
            df_usa['beta'] = pd.to_numeric(df_usa['beta'], errors='coerce')
            self.dataframe = df_usa
            print("Data loaded and prepared successfully.")
            return True
        except FileNotFoundError:
            print(f"Error: The file '{self.file_path}' was not found.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred during data loading: {e}")
            return False

    def calculate_average_beta(self) -> pd.DataFrame:
        """
        Calculates the average beta (risk profile) for companies in each state.
        """
        print("Calculating average beta by state...")
        df_clean = self.dataframe.dropna(subset=['beta', 'hq_state'])
        beta_df = df_clean.groupby('hq_state')['beta'].mean().reset_index()
        return beta_df

    def generate_map(self, data_df: pd.DataFrame, data_column: str, title: str, labels: dict,
                     output_filename: str, color_scale: str = "Viridis", color_midpoint: float = None,
                     custom_hover_col: str = None):
        """
        Generates and saves an interactive choropleth map visualization with improved hover info.
        """
        print(f"Generating map for '{data_column}', saving to '{output_filename}'...")

        hover_template = '<b>%{hovertext}</b><br>'
        if custom_hover_col:
            # Use custom text column for the hover value
            hover_template += f'{labels.get(data_column, data_column)}: %{{customdata[0]}}<extra></extra>'
        else:
            # Default to formatting the numeric value
            hover_template += f'{labels.get(data_column, data_column)}: %{{z:.3f}}<extra></extra>'

        fig = px.choropleth(
            data_df,
            locations='hq_state',
            locationmode="USA-states",
            color=data_column,
            scope="usa",
            color_continuous_scale=color_scale,
            color_continuous_midpoint=color_midpoint,
            labels=labels,
            title=title,
            hover_name='state_name',
            custom_data=[custom_hover_col] if custom_hover_col else None
        )

        # Update the hover template
        fig.update_traces(hovertemplate=hover_template)

        fig.update_layout(title_x=0.5, geo=dict(bgcolor='rgba(0,0,0,0)', lakecolor='rgba(0,0,0,0)'))
        fig.write_html(output_filename)
        print("Map generation complete.")

    def generate_correlation_plot(self, data_df: pd.DataFrame, x_col: str, y_col: str, title: str,
                                  labels: dict, output_filename: str, text_col: str = None):
        """
        Generates a scatter plot to visualize correlation between two variables.
        """
        print(
            f"Generating correlation plot between '{x_col}' and '{y_col}', saving to '{output_filename}'...")
        fig = px.scatter(
            data_df,
            x=x_col,
            y=y_col,
            text=text_col,
            title=title,
            labels=labels,
            trendline="ols",  # Add an Ordinary Least Squares regression line
        )
        if text_col:
            fig.update_traces(textposition='top center')
        fig.update_layout(title_x=0.5)
        fig.write_html(output_filename)
        print("Correlation plot generation complete.")


def format_leaning(margin):
    """
    Formats the political leaning margin for hover text.
    Positive margins indicate Republican leaning, negative indicate Democratic leaning.
    """
    if margin > 0:
        return f"{abs(margin):.2f}% Republican"
    else:
        return f"{abs(margin):.2f}% Democratic"


def plot_state_tax_rates(visualizer: GeoVisualizer, state_names_df: pd.DataFrame, output_dir: str):
    """
    Plots the state corporate tax rates.
    """
    print("\n--- Generating Tax Rate Map ---")
    tax_df = pd.DataFrame(list(STATE_CORPORATE_TAX_RATES.items()), columns=['hq_state', 'tax_rate'])
    tax_df = pd.merge(tax_df, state_names_df, on='hq_state')
    visualizer.generate_map(
        data_df=tax_df,
        data_column='tax_rate',
        title='State Corporate Tax Rates (2024)',
        labels={'tax_rate': 'Tax Rate'},
        output_filename=os.path.join(output_dir, 'tax_rate_map.html')
    )
    return tax_df


def plot_state_political_leaning(visualizer: GeoVisualizer, state_names_df: pd.DataFrame, output_dir: str):
    """
    Plots the state political leaning using a diverging color scale.
    """
    print("\n--- Generating Political Leaning Map (Red/Blue) ---")
    politics_df = pd.DataFrame(list(STATE_POLITICAL_LEANING_2024.items()),
                               columns=['hq_state', 'rep_margin_2024'])

    politics_df['political_label'] = politics_df['rep_margin_2024'].apply(format_leaning)

    politics_df = pd.merge(politics_df, state_names_df, on='hq_state')
    visualizer.generate_map(
        data_df=politics_df, data_column='rep_margin_2024',
        title='State Political Leaning (Red = Republican, Blue = Democratic)',
        labels={'rep_margin_2024': 'Political Leaning'},
        output_filename=os.path.join(output_dir, 'geo_political_leaning.html'),
        color_scale='RdBu_r',
        color_midpoint=0,
        custom_hover_col='political_label'
    )
    return politics_df


def plot_average_beta(visualizer: GeoVisualizer, state_names_df: pd.DataFrame, output_dir: str):
    """
    Plots the average company beta by state.
    """
    print("\n--- Generating Average Beta Map ---")
    beta_df = visualizer.calculate_average_beta()
    beta_df = pd.merge(beta_df, state_names_df, on='hq_state')
    visualizer.generate_map(
        data_df=beta_df,
        data_column='beta',
        title='Average Company Risk Profile (Beta) by State',
        labels={'beta': 'Average Beta'},
        output_filename=os.path.join(output_dir, 'geo_average_beta.html')
    )
    return beta_df


def plot_correlation(visualizer: GeoVisualizer, beta_df: pd.DataFrame, politics_df: pd.DataFrame,
                     output_dir: str):
    """
    Plots the correlation between state political leaning and average company beta.
    """
    print("\n--- Generating Correlation Plot (Politics vs. Beta) ---")
    correlation_df = pd.merge(beta_df, politics_df, on='hq_state')
    visualizer.generate_correlation_plot(
        data_df=correlation_df, x_col='rep_margin_2024', y_col='beta',
        title='Correlation between State Politics and Company Risk Profile',
        labels={'rep_margin_2024': 'Political Leaning (Rep > 0 > Dem)', 'beta': 'Average Company Beta'},
        output_filename=os.path.join(output_dir, 'correlation_plot.html'),
        text_col='hq_state'
    )
    return correlation_df


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.normpath(os.path.join(script_dir, '..', '..', 'data'))
    output_dir = os.path.normpath(os.path.join(data_dir, 'geo_visualizes'))
    os.makedirs(output_dir, exist_ok=True)
    company_data_file = os.path.join(data_dir, 'company_info.csv')

    visualizer = GeoVisualizer(company_data_file)
    state_names_df = pd.DataFrame(list(STATE_NAMES.items()), columns=['hq_state', 'state_name'])

    plot_state_tax_rates(visualizer, state_names_df, output_dir)
    politics_df = plot_state_political_leaning(visualizer, state_names_df, output_dir)

    visualizer.load_and_prepare_data()

    beta_df = plot_average_beta(visualizer, state_names_df, output_dir)
    plot_correlation(visualizer, beta_df, politics_df, output_dir)


if __name__ == "__main__":
    main()
