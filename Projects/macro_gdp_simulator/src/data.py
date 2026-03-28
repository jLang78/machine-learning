# src/data.py
import os
import sys
import wbgapi as wb
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def fetch_live_economic_data(output_dir="data/raw"):
    """
    I connect to the World Bank API to download recent macroeconomic data.
    I target a stable recent year because MRV (Most Recent Value) often
    returns unpublished blanks for lagging indicators like GDP.
    """
    os.makedirs(os.path.join(project_root, output_dir), exist_ok=True)

    indicators = {
        'NY.GDP.PCAP.CD': 'gdp_per_capita',
        'NE.TRD.GNFS.ZS': 'trade_openness',
        'BX.KLT.DINV.WD.GD.ZS': 'fdi_inflows',
        'NE.GDI.TOTL.ZS': 'capital_formation',
        'SP.DYN.LE00.IN': 'life_expectancy',
        'SL.TLF.CACT.ZS': 'labor_force_part',
        'SP.POP.GROW': 'population_growth',
        'FP.CPI.TOTL.ZG': 'inflation_rate'
    }

    print("Initiating live API request to the World Bank...")

    # I query a recent stable year to guarantee global data completeness.
    stable_year = 2022

    df_raw = wb.data.DataFrame(indicators.keys(), time=stable_year, labels=True)

    df_raw = df_raw.reset_index()
    df_raw = df_raw.rename(columns=indicators)
    df_raw = df_raw.rename(columns={'economy': 'country_code', 'Country': 'country_name'})

    raw_path = os.path.join(project_root, output_dir, "world_bank_live_raw.csv")
    df_raw.to_csv(raw_path, index=False)

    print(f"Success. I fetched stable data for {len(df_raw)} economies.")
    print(f"Raw data saved to {raw_path}")


if __name__ == "__main__":
    fetch_live_economic_data()