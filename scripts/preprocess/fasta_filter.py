import pandas as pd

df1 = pd.read_csv('data/raw/iedb_positives.csv')
df2 = pd.read_csv('data/raw/vdjdb_positives.csv')

frames = [df1, df2]
df3 = pd.concat(frames)

HLA_list = list(set(df1['HLA'].unique()).union(df2['HLA'].unique()))
HLA_list = [hla.replace('HLA-', '') for hla in HLA_list] # remove HLA- prefix to compare with fasta file

with open('data/raw/HLA/hla_list.txt', 'w') as f:
    for hla in HLA_list:
        f.write(f"{hla}\n")

# test for one HLA
#HLA_list = ['HLA-A*01:01']
HLA_list = [hla.replace('HLA-', '') for hla in HLA_list]  # Ensure no 'HLA-' prefix

filtered_records = []
header = []
sequence_list = []

with open('data/raw/HLA/hla_prot.fasta', 'r') as fasta_file:
    fasta_lines = fasta_file.readlines()

for hla in HLA_list:
    found = False
    record = ''
    sequence = '' #needs to be empty for each HLA
    for i, line in enumerate(fasta_lines):
        if line.startswith('>') and 'N ' not in line:
            if found:
                break
            if hla in line:
                # Replace header with just the HLA tag
                record = f'>{hla}\n'
                # Collect the sequence lines
                for seq_line in fasta_lines[i+1:]:
                    if seq_line.startswith('>'):
                        break
                    record += seq_line
                    sequence += seq_line
                filtered_records.append(record)
                header.append(f'HLA-{hla}')
                sequence_list.append(sequence)
                found = True
    # Optionally, warn if not found

# Write filtered records to a new file
with open('data/raw/HLA/hla_prot_filtered_firsthits.fasta', 'w') as out_file:
    out_file.writelines(filtered_records)

df_filtered = pd.DataFrame({'HLA': header, 'HLA_sequence': sequence_list})
df_merge = df3.merge(df_filtered, on='HLA', how='left')
#print(df_merge.head())
df_merge.to_csv('data/raw/HLA/full_positives_hla_seq.csv', index=False)

# Count headers in output file and compare with HLA list
with open('data/raw/HLA/hla_prot_filtered_firsthits.fasta', 'r') as f:
    header_count = sum(1 for line in f if line.startswith('>'))
print(f"Number of HLAs in list: {len(HLA_list)}")
print(f"Number of sequences written: {header_count}")

# Find missing HLAs by comparing sets
written_hlas = set()
with open('data/raw/HLA/hla_prot_filtered_firsthits.fasta', 'r') as f:
    for line in f:
        if line.startswith('>'):
            written_hlas.add(line[1:].strip())

missing_hlas = set(HLA_list) - written_hlas
print("\nMissing HLAs:")
for hla in missing_hlas:
    print(hla)

# # if there are missing HLAs, check the A_prot.fasta fasta file for the missing HLAs
# filtered_records = []
# header = []
# sequence_list = []

# with open('data/raw/HLA/A_prot.fasta', 'r') as fasta_file:
#     fasta_lines = fasta_file.readlines()

# for hla in missing_hlas:
#     found = False
#     record = ''
#     sequence = '' #needs to be empty for each HLA
#     for i, line in enumerate(fasta_lines):
#         if line.startswith('>') and 'N ' not in line:
#             if found:
#                 break
#             if hla in line:
#                 # Replace header with just the HLA tag
#                 record = f'>{hla}\n'
#                 # Collect the sequence lines
#                 for seq_line in fasta_lines[i+1:]:
#                     if seq_line.startswith('>'):
#                         break
#                     record += seq_line
#                     sequence += seq_line
#                 filtered_records.append(record)
#                 header.append(f'HLA-{hla}')
#                 sequence_list.append(sequence)
#                 found = True
#     # Optionally, warn if not found

# df_missing_hlas = pd.DataFrame({'HLA': header, 'Sequence': sequence_list})
# df_all = df_merge.merge(df_missing_hlas, on='HLA', how='left')
# #print(df_merge.head())
# df_all.to_csv('data/raw/HLA/full_positives_hla_seq.csv', index=False)
