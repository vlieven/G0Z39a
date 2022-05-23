from pathlib import Path
from typing import Final, Mapping

import pandas as pd
import requests

from .base import Dataset


class Vaccinations(Dataset):
    """Class which represents the vaccinations dataset provided by the CDC.
    Source: https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh
    """

    url: Final[str] = "https://data.cdc.gov/api/views/8xkx-amqh/rows.csv?accessType=DOWNLOAD"
    destination: Final[Path] = Dataset.ROOT_DIR / "raw" / "vaccinations"

    @classmethod
    def _filename(cls) -> str:
        return "cdc_vaccinations.csv"

    @classmethod
    def _filepath(cls) -> Path:
        return cls.destination / cls._filename()

    @classmethod
    def collected(cls) -> bool:
        return cls._filepath().is_file()

    @classmethod
    def _collect_data(cls) -> None:
        file: str = cls._filename()
        with requests.get(cls.url, stream=True) as response:
            cls._store_response(
                response, folder=cls.destination, file=file, description="Vaccination data"
            )

    @classmethod
    def _load_dataframe(cls) -> pd.DataFrame:
        return pd.read_csv(cls._filepath(), dtype=cls._schema(), parse_dates=["Date"])

    @classmethod
    def _schema(cls) -> Mapping[str, str]:
        return {
            "FIPS": "string",
            "MMWR_week": "UInt8",
            "Recip_County": "string",
            "Recip_State": "string",
            "Completeness_pct": "Float32",
            "Administered_Dose1_Recip": "Float32",
            "Administered_Dose1_Pop_Pct": "Float32",
            "Administered_Dose1_Recip_5Plus": "Float32",
            "Administered_Dose1_Recip_5PlusPop_Pct": "Float32",
            "Administered_Dose1_Recip_12Plus": "Float32",
            "Administered_Dose1_Recip_12PlusPop_Pct": "Float32",
            "Administered_Dose1_Recip_18Plus": "Float32",
            "Administered_Dose1_Recip_18PlusPop_Pct": "Float32",
            "Administered_Dose1_Recip_65Plus": "Float32",
            "Administered_Dose1_Recip_65PlusPop_Pct": "Float32",
            "Series_Complete_Yes": "Float32",
            "Series_Complete_Pop_Pct": "Float32",
            "Series_Complete_5Plus": "Float32",
            "Series_Complete_5PlusPop_Pct": "Float32",
            "Series_Complete_5to17": "Float32",
            "Series_Complete_5to17Pop_Pct": "Float32",
            "Series_Complete_12Plus": "Float32",
            "Series_Complete_12PlusPop_Pct": "Float32",
            "Series_Complete_18Plus": "Float32",
            "Series_Complete_18PlusPop_Pct": "Float32",
            "Series_Complete_65Plus": "Float32",
            "Series_Complete_65PlusPop_Pct": "Float32",
            "Booster_Doses": "Float32",
            "Booster_Doses_Vax_Pct": "Float32",
            "Booster_Doses_12Plus": "Float32",
            "Booster_Doses_12Plus_Vax_Pct": "Float32",
            "Booster_Doses_18Plus": "Float32",
            "Booster_Doses_18Plus_Vax_Pct": "Float32",
            "Booster_Doses_50Plus": "Float32",
            "Booster_Doses_50Plus_Vax_Pct": "Float32",
            "Booster_Doses_65Plus": "Float32",
            "Booster_Doses_65Plus_Vax_Pct": "Float32",
            "SVI_CTGY": "category",
            "Series_Complete_Pop_Pct_SVI": "Float32",
            "Series_Complete_5PlusPop_Pct_SVI": "Float32",
            "Series_Complete_5to17Pop_Pct_SVI": "Float32",
            "Series_Complete_12PlusPop_Pct_SVI": "Float32",
            "Series_Complete_18PlusPop_Pct_SVI": "Float32",
            "Series_Complete_65PlusPop_Pct_SVI": "Float32",
            "Metro_status": "category",
            "Series_Complete_Pop_Pct_UR_Equity": "Float32",
            "Series_Complete_5PlusPop_Pct_UR_Equity": "Float32",
            "Series_Complete_5to17Pop_Pct_UR_Equity": "Float32",
            "Series_Complete_12PlusPop_Pct_UR_Equity": "Float32",
            "Series_Complete_18PlusPop_Pct_UR_Equity": "Float32",
            "Series_Complete_65PlusPop_Pct_UR_Equity": "Float32",
            "Booster_Doses_Vax_Pct_SVI": "Float32",
            "Booster_Doses_12PlusVax_Pct_SVI": "Float32",
            "Booster_Doses_18PlusVax_Pct_SVI": "Float32",
            "Booster_Doses_65PlusVax_Pct_SVI": "Float32",
            "Booster_Doses_Vax_Pct_UR_Equity": "Float32",
            "Booster_Doses_12PlusVax_Pct_UR_Equity": "Float32",
            "Booster_Doses_18PlusVax_Pct_UR_Equity": "Float32",
            "Booster_Doses_65PlusVax_Pct_UR_Equity": "Float32",
            "Census2019": "UInt64",
            "Census2019_5PlusPop": "UInt64",
            "Census2019_5to17Pop": "UInt64",
            "Census2019_12PlusPop": "UInt64",
            "Census2019_18PlusPop": "UInt64",
            "Census2019_65PlusPop": "UInt64",
        }
