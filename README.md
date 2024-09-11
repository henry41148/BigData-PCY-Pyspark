# PCY Algorithm Implementation in PySpark

## Project Overview
This project implements the **PCY (Park-Chen-Yu) Algorithm** using PySpark to analyze large-scale shopping basket data. The goal is to find frequent item pairs and generate association rules based on the provided transaction dataset.

### Key Features:
- **Data Input**: Reads shopping basket data from `baskets.csv`.
- **Frequent Itemsets**: Mines frequent item pairs using the PCY algorithm.
- **Association Rules**: Generates association rules from the frequent item pairs.
- **Output**: Results are saved as CSV files (`pcy_frequent_pairs.csv` and `pcy_association_rules.csv`).

## Files and Methods
- **baskets.csv**: Input file containing shopping transaction data.
- **PCY Class**:
  - `__init__(path, s, c)`: Initializes the algorithm with the file path, support threshold (`s`), and confidence threshold (`c`).
  - `run()`: Executes the PCY algorithm and saves the output CSV files.

### Example Structure of `baskets.csv`:
- `Member_number`: Customer ID
- `Date`: Purchase date in `dd/mm/yyyy` format
- `itemDescription`: Name of the purchased item
- `year`, `month`, `day`, `day_of_week`: Purchase details

## How to Set Up the Environment
1. **Download and Install Apache Spark**:
   You can download the Spark binary using the `wget` command:
   ```bash
   !wget -q http://archive.apache.org/dist/spark/spark-3.1.1/spark-3.1.1-bin-hadoop3.2.tgz
   !tar xf spark-3.1.1-bin-hadoop3.2.tgz
   Or download it manually from [Apache Spark Archive](http://archive.apache.org/dist/spark/spark-3.1.1/).

2. **Install PySpark**:
   You can install `findspark` to help with Spark integration in your Python environment:
   ```bash
   !pip install -q findspark

3. **Download the Basket Data**:
   Ensure the `baskets.csv` file is available in your directory. For example:
   ```python
   path = "/content/drive/MyDrive/baskets.csv"

## How to Run the Project
1. Clone the repository.
2. Set up the environment as described above.
3. Modify the input path in the code to point to your `baskets.csv` file.
4. Run the script using:
   ```bash
   spark-submit pcy_algorithm.py

## Output Files
- `pcy_frequent_pairs.csv`: Contains frequent item pairs based on the support threshold.
- `pcy_association_rules.csv`: Contains association rules generated from the frequent item pairs.

## Requirements
- PySpark
- Python 3.x

## Notes
- The project strictly follows big data principles, avoiding in-memory operations for large datasets.
- No pre-built PCY libraries were used, as per the project requirements.

## License
This project is licensed under the MIT License.
