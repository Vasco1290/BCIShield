import os
import sys

def print_instructions():
    """
    Prints instructions for downloading the BCI Competition IV Dataset 2a.
    """
    instructions = f"""
    ================================================================================
    BCI Competition IV Dataset 2a Download Helper
    ================================================================================
    The dataset is publicly available at: 
    https://www.bbci.de/competition/iv/

    To download the .gdf files:
    1. Visit the BNCI Horizon 2020 repository or the official BBCI website.
       Alternatively, you can find the dataset on Zenodo or through MOABB.
    2. Download the A01T.gdf to A09T.gdf (Training) and A01E.gdf to A09E.gdf (Evaluation).
    3. Place the .gdf files in the following directory:
       {os.path.abspath(os.path.join(os.path.dirname(__file__), 'raw'))}
       
    Example file structure after download:
    BCIShield/
    └── data/
        └── raw/
            ├── A01T.gdf
            ├── A01E.gdf
            ├── A02T.gdf
            └── ...

    Once downloaded, the dataset.py module will handle the parsing and preprocessing.
    ================================================================================
    """
    print(instructions)

if __name__ == "__main__":
    print_instructions()
    print("Please follow the instructions above to acquire the dataset.")
