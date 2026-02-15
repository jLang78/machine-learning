
import csv
import os


def load_csv(filename, target_column_index=-1, conversion_func=float):
    """
    Loads a CSV into a list of features (X) and a list of labels (y).

    Args:
        filename (str): Name of the file inside the 'data' folder.
        target_column_index (int): The column index of the label (y).
                                   -1 means the last column.
        conversion_func (func): Function to convert strings to numbers
                                (e.g., float or int).

    Returns:
        X (list of lists): The feature matrix.
        y (list): The target vector.
    """

    # Constructing the full path to the data folder
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(base_dir, 'data', filename)

    X = []
    y = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skipping the header row

            for row in reader:
                if not row: continue  # Skipping empty rows

                # Extracting y (label)
                if target_column_index == -1:
                    label_val = row[-1]
                    features = row[:-1]
                else:
                    label_val = row[target_column_index]
                    features = row[:target_column_index] + row[target_column_index + 1:]

                # Converting features to numbers
                try:
                    converted_features = [conversion_func(x) for x in features]

                    # Converting label (if numeric, convert...if string, keep)
                    try:
                        converted_label = float(label_val)
                    except ValueError:
                        converted_label = label_val

                    X.append(converted_features)
                    y.append(converted_label)

                except ValueError:
                    print(f"Skipping malformed row: {row}")
                    continue

        return X, y

    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found in {base_dir}/data/")
        return [], []