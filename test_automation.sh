# batch_process.sh
#!/bin/bash

# Check if names file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 competition_list.txt"
    exit 1
fi

# Read the names file
competition_list=$1

# Check if the names file exists
if [ ! -f "$competition_list" ]; then
    echo "Error: Competitions file '$competition_list' not found!"
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Read each name from the file and process it twice
while IFS= read -r name; do
    echo "Processing: $name"
    
    # First run
    echo "Starting first run for $name"
    python3 run_raagul.py "$name.md" > "logs/${name}.log" 2>&1
    echo "Completed first run for $name"
    
    # Second run
    echo "Starting second run for $name"
    python3 run_raagul.py "$name.md" > "logs/${name}_2.log" 2>&1
    echo "Completed second run for $name"
    
done < "$competition_list"

echo "All processing completed!"
