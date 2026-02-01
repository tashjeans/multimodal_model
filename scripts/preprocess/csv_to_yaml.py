import pandas as pd
import yaml
from pathlib import Path

# def load_and_concatenate(csv_paths):
#     dataframes = [pd.read_csv(path) for path in csv_paths]
#     return pd.concat(dataframes, ignore_index=True)
# think there needs to be one fasta file for each complex

def convert_to_structural_boltz_yaml(df, output_path):
    targets = []
    for idx, row in df.iterrows():
        peptide = row.get('Peptide')
        mhc = row.get('HLA_sequence')
        # Clean HLA sequence by removing line breaks and commas
        mhc = str(mhc).replace('\n', '').replace(',', '').strip() if pd.notnull(mhc) else "X"
        tcra = row['TCRa'] if pd.notnull(row.get('TCRa')) else "X"
        tcrb = row['TCRb'] if pd.notnull(row.get('TCRb')) else "X"

        if pd.notnull(peptide) and pd.notnull(mhc):
            target = {
                'name': f"example_{idx}",
                'sequences': [
                    {'protein': {'id': 'M', 'sequence': mhc, 'msa': 'empty'}},
                    {'protein': {'id': 'P', 'sequence': peptide, 'msa': 'empty'}},
                    {'protein': {'id': 'A', 'sequence': tcra, 'msa': 'empty'}},
                    {'protein': {'id': 'B', 'sequence': tcrb, 'msa': 'empty'}},
                ],
            }
            targets.append(target)

    yaml_obj = {
        'version': 1,
        'targets': targets
    }

    with open(output_path, 'w') as f:
        yaml.dump(yaml_obj, f, default_flow_style=False, sort_keys=False)
    print(f"Saved structural Boltz YAML to: {output_path}")


if __name__ == "__main__":
    # Define paths
    raw_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "HLA"
    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # csv_paths = [
    #     raw_dir / "vdjdb_positives.csv",
    #     raw_dir / "iedb_positives.csv"
    # ]
    output_yaml = processed_dir / "data_for_boltz_attempt1.yaml"

    # Run script
    # df = load_and_concatenate(csv_paths)
    df = pd.read_csv(raw_dir / "full_positives_hla_seq.csv")
    convert_to_structural_boltz_yaml(df, output_yaml)
