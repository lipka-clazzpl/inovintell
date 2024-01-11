"""
Author: Przemysław Lipka
Project: Diabetic Macular Oedema Treatment Response Prediction
Date: 11 January 2024

This code is the intellectual property of Przemysław Lipka. It is designed
for the purpose of research and analysis in a non-commercial setting.
Unauthorized use, distribution, or replication for commercial purposes is
strictly prohibited.

There are four primary data sources available
    THxBaseline: Patient demographics, medical history, and previous eye treatments.
    PtRoster: Information on which eye is the study eye for each patient.
    TTrtGroupUnmasked: Details of the treatment assigned to each patient.
    TVAReVATest.csv: Visual acuity scores of the patients.

This pipeline step will 
    - merge datasets on PtID (Patient ID)
    - filter out irrelevant data (won't analyze full time series, only final result after 52 weeks)
    - filter out irrelevant columns

"""

from datetime import datetime
from pathlib import Path
import pandas as pd
import fire


def _filter_data_for_study_eyes(df) -> pd.DataFrame:
    """Filter out data for non-study eyes. Limit to baseline and after one year evaluations."""

    filtered = df[df["Visit"].isin(["Baseline", "52 week"])]

    filtered["EtdrsScore"] = filtered.apply(
        lambda x: x["EtdrsOD"] if x["StudyEye"] == "OD" else x["EtdrsOS"], axis=1
    )
    return filtered


def main(colunns_to_keep: list = []):
    # All artifacts are tagged with this timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Load data sources
    roster = pd.read_csv("data/raw/PtRoster.txt", sep="|")
    treatment_group = pd.read_csv("data/raw/TTrtGroupUnmasked.txt", sep="|")
    baseline = pd.read_csv("data/raw/THxBaseline.txt", sep="|")

    # Target needs a little bit of ❤️
    target = pd.read_csv("data/raw/TVAReVATest.csv")

    target = _filter_data_for_study_eyes(target.merge(roster, on="PtID"))

    response_feature = _enrich_with_response(roster, target)

    # Merge with patient data
    merged_data = baseline.merge(response_feature, on="PtID", how="left")
    merged_data = merged_data.merge(treatment_group, on="PtID")

    # Additional filtering of the columns for experimentation
    df_to_save = merged_data[colunns_to_keep] if colunns_to_keep else merged_data

    filename_with_timestamp = f"merged_data_{timestamp}.csv"

    df_to_save.to_csv(Path("data/merged") / filename_with_timestamp, index=False)


def _enrich_with_response(roster, target_filtered):
    """
    Enriches the given roster DataFrame with the response to treatment information.

    Target variable: Response to treatment (Increase in ETDRS score by 10 or more)

    Args:
        roster (pandas.DataFrame): The roster DataFrame containing patient information.
        target_filtered (pandas.DataFrame): The filtered target DataFrame containing ETDRS scores.

    Returns:
        pandas.DataFrame: The enriched roster DataFrame with the response to treatment information.
    """
    enriched_roster = target_filtered.pivot(
        index="PtID", columns="Visit", values="EtdrsScore"
    )

    enriched_roster.columns = [
        "EtdrsScore_52_week",
        "EtdrsScore_Baseline",
    ]
    enriched_roster = enriched_roster.reset_index()
    enriched_roster = pd.merge(roster, enriched_roster, on="PtID", how="left")

    enriched_roster["Response"] = (
        enriched_roster["EtdrsScore_52_week"] - enriched_roster["EtdrsScore_Baseline"]
    ) >= 10

    return enriched_roster


if __name__ == "__main__":
    fire.Fire(main)
