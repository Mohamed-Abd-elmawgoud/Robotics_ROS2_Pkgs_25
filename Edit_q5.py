#!/usr/bin/env python3
import csv

def reverse_trajectory(input_file, output_file):
    """
    Reverse a trajectory CSV file from last step to first step
    Renumbers the steps from 0 to N-1
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to output CSV file
    """
    rows = []
    
    # Read the CSV file
    with open(input_file, 'r') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        
        for row in reader:
            rows.append(row)
    
    # Reverse the order of rows (last to first)
    rows.reverse()
    
    # Renumber the steps from 0 to N-1
    for i, row in enumerate(rows):
        row['step'] = str(i)
    
    # Write the reversed data to output file
    with open(output_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"Successfully reversed {len(rows)} rows")
    print(f"Original order: step 0 -> step {len(rows)-1}")
    print(f"Reversed order: step {len(rows)-1} -> step 0")
    print(f"Output written to: {output_file}")


if __name__ == '__main__':
    # Modify these filenames as needed
    input_filename = 'Pick.csv'
    output_filename = 'Pick_reversed.csv'
    
    reverse_trajectory(input_filename, output_filename)