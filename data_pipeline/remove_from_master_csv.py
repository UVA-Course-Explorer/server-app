import csv

# Specify the input and output CSV file names
input_file_name = 'combined.csv'
output_file_name = 'combined_mod.csv'

# Specify the column name you want to check, and the value to remove
column_name = 'strm'
values_to_remove = ['1248', '1246']

# Read the original CSV, filter rows, and write to a new CSV
def filter_csv(input_file_name, output_file_name, column_name, value_to_remove):
    with open(input_file_name, mode='r', newline='') as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = reader.fieldnames
        
        # Ensure the column to check exists in the CSV
        if column_name not in fieldnames:
            print(f"Error: The column '{column_name}' was not found in the input file.")
            return
        
        with open(output_file_name, mode='w', newline='') as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                if row[column_name] not in values_to_remove:
                    writer.writerow(row)

# Call the function with the specified parameters
filter_csv(input_file_name, output_file_name, column_name, values_to_remove)

print(f"Rows with {column_name} not equal to '{values_to_remove}' have been written to {output_file_name}")
