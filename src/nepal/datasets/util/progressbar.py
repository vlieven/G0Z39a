from typing import Optional

from requests import Response
from tqdm.auto import tqdm


def download(response: Response, description: Optional[str] = None) -> tqdm:
    return tqdm(desc=description, total=_file_size(response), unit="B", unit_scale=True)


def _file_size(response: Response) -> Optional[int]:
    return int(response.headers.get("Content-Length", 0)) or None
