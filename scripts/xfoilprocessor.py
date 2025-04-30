import subprocess
import os
import csv
from itertools import product
import psutil

# normalise_geometry function to ensure consistent geometry data
# This function takes a list of (x, y) tuples and normalises it to a specified number of points.
def normalise_geometry(geometry_data, num_points=100):
    if not geometry_data:
        return []
    n = len(geometry_data)
    if n == num_points:
        return geometry_data.copy()
    normalized = []
    for i in range(num_points):
        pos = (i / (num_points - 1)) * (n - 1)
        lower = int(pos)
        upper = lower + 1
        if upper >= n:
            upper = lower
        frac = pos - lower
        x_lower, y_lower = geometry_data[lower]
        x_upper, y_upper = geometry_data[upper]
        x = x_lower * (1 - frac) + x_upper * frac
        y = y_lower * (1 - frac) + y_upper * frac
        normalized.append((x, y))
    return normalized

# This function runs XFOIL with the specified parameters and returns the results.
# It handles the input and output files, as well as any errors that may occur during execution.
def run_xfoil(aerofoil, ncrit, reynolds, mach, timeout=10):
    input_filename = "xfoil_input.txt"
    output_filename = f"xfoil_output_{ncrit}_{reynolds}_{mach}.txt"
    success = False
    result_data = {}
    proc = None

    with open(input_filename, 'w') as f:
        f.write(f"LOAD {aerofoil}\n")
        f.write("PANE\n")
        f.write("OPER\n")
        f.write("Iter\n")
        f.write('250\n')
        f.write("VISC\n")
        f.write(f"{reynolds}\n")
        f.write("MACH\n")
        f.write(f"{mach}\n") 
        f.write(f"N {ncrit}\n")
        f.write("PACC\n")
        f.write(f"{output_filename}\n")
        f.write("\n")
        f.write("ASEQ -5 15 5\n")
        f.write("PACC OFF\n")
        f.write("\nQUIT\n")

    try:
        with open(input_filename, 'r') as input_file:
            proc = subprocess.Popen(
                [r"C:\Program Files\XFOIL\xfoil.exe"], #replace with path to  XFOIL executable
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                text=True
            )
            stdout, stderr = proc.communicate(timeout=timeout)
            if proc.returncode != 0:
                raise subprocess.CalledProcessError(proc.returncode, 'xfoil.exe', output=stdout, stderr=stderr)

        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            with open(output_filename, 'r') as output_file:
                lines = output_file.readlines()
                parsed_data = parse_result_data(lines)
                if parsed_data:
                    success = True
                    result_data['parsed_data'] = parsed_data
                else:
                    print(f"No converged data for {aerofoil}, Ncrit={ncrit}, Re={reynolds}, Mach={mach}.")
        else:
            print(f"No output file generated for {aerofoil}, Ncrit={ncrit}, Re={reynolds}, Mach={mach}.")

    except subprocess.TimeoutExpired:
        print(f"XFOIL timed out for {aerofoil}, Ncrit={ncrit}, Re={reynolds}, Mach={mach}.")
        if proc:
            # Kill the process and its children
            parent = psutil.Process(proc.pid)
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
            stdout, stderr = proc.communicate()

    except subprocess.CalledProcessError as e:
        print(f"XFOIL failed for {aerofoil}, Ncrit={ncrit}, Re={reynolds}, Mach={mach}: {e}")
    except Exception as e:
        print(f"Unexpected error for {aerofoil}, Ncrit={ncrit}, Re={reynolds}, Mach={mach}: {e}")
    finally:
        if os.path.exists(input_filename):
            os.remove(input_filename)
        if os.path.exists(output_filename):
            try:
                os.remove(output_filename)
            except PermissionError:
                print(f"Could not delete {output_filename}, it may be locked.")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    return success, result_data

# This function parses the result data from the XFOIL output file.
# It extracts the relevant aerodynamic coefficients and other parameters.
def parse_result_data(result_lines):
    parsed_data = []
    table_start = False

    for line in result_lines:
        if "alpha" in line.lower() and "CL" in line:
            table_start = True
            continue
        if table_start:
            try:
                values = list(map(float, line.split()))
                if len(values) >= 7:
                    parsed_data.append(values[:7])
            except ValueError:
                continue
    return parsed_data

def extract_geometry_data(dat_file):
    geometry_data = []
    with open(dat_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 2:
            try:
                x = float(parts[0])
                y = float(parts[1])
                geometry_data.append((x, y))
            except ValueError:
                continue

    return geometry_data

# This function loads existing data from the input and error CSV files.
# It returns sets of existing entries and errors to avoid reprocessing them.
def load_existing_data(inp_csv, err_csv):
    existing_entries = set()
    existing_errors = set()

    if os.path.exists(inp_csv):
        with open(inp_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                try:
                    airfoil_num = int(row[0])
                    ncrit = float(row[-4])
                    re = float(row[-3])
                    mach = float(row[-2])
                    alpha = float(row[-1])
                    existing_entries.add((airfoil_num, ncrit, re, mach, alpha))
                except (IndexError, ValueError):
                    continue

    if os.path.exists(err_csv):
        with open(err_csv, 'r') as f:
            reader = csv.reader(f)
            next(reader)  
            for row in reader:
                try:
                    airfoil_num = int(row[0])
                    ncrit = float(row[1])
                    mach = float(row[2])
                    re = float(row[3])
                    existing_errors.add((airfoil_num, ncrit, mach, re))
                except (IndexError, ValueError):
                    continue

    return existing_entries, existing_errors

# Configuration
folder_path = r"data\uiuc_airfoils_dat"
os.makedirs("results", exist_ok=True)

# Create CSV files if they do not exist
# CSV file paths for input parameters, output parameters, error parameters and airfoil legend
inp_param_csv = r"data\xfoil_results\inp_param.csv"
out_param_csv = r"data\xfoil_results\out_param.csv"
err_param_csv = r"data\xfoil_results\err_param.csv"
air_leg_csv = r"data\xfoil_results\air_leg.csv"

csv_config = [
    (inp_param_csv, ["airfoil_num"] + [f"{coord}{i}" for i in range(1, 101) for coord in ("X", "Y")] + ["Ncrit", "Reynolds", "Mach", "Alpha"]),
    (out_param_csv, ["airfoil_num", "CL", "CD", "CM", "CDp", "Top_xtr", "Bot_xtr"]),
    (err_param_csv, ["airfoil_num", "Ncrit", "Mach", "Reynolds", "alphalst"]),
    (air_leg_csv, ["airfoil_num", "airfoil_name"])
]

for csv_file, header in csv_config:
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            csv.writer(f).writerow(header)

airfoil_dict = {}
max_num = 0

# Read existing airfoil numbers from the CSV file
if os.path.exists(air_leg_csv):
    with open(air_leg_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) >= 2:
                num = int(row[0])
                name = row[1]
                airfoil_dict[name] = num
                if num > max_num:
                    max_num = num

existing_entries, existing_errors = load_existing_data(inp_param_csv, err_param_csv)

# Define the parameters for the XFOIL runs
ncrit_values = [5, 7, 9, 11]
reynolds_values = [1e4, 5e4, 1e5, 5e5, 1e6]
mach_values = [0.1, 0.2, 0.3]

# Process airfoils
for filename in os.listdir(folder_path):
    input_file = os.path.join(folder_path, filename)
    if os.path.isfile(input_file) and filename.lower().endswith('.dat'):
        airfoil_name = os.path.splitext(filename)[0]

        # Check if the airfoil is already registered
        if airfoil_name not in airfoil_dict:
            max_num += 1
            airfoil_dict[airfoil_name] = max_num
            
            with open(air_leg_csv, 'a', newline='') as f:
                csv.writer(f).writerow([max_num, airfoil_name])
            print(f"Registered new airfoil: {airfoil_name} (ID: {max_num})")
        
        airfoil_num = airfoil_dict[airfoil_name]
        geometry_data = extract_geometry_data(input_file)
        if not geometry_data:
            print(f"Skipping {airfoil_name} due to empty geometry data.")
            continue
        
        # Check if the geometry data is valid
        y_coords = [y for x, y in geometry_data]
        thickness_ratio = max(y_coords) - min(y_coords)
        param_combinations = product(ncrit_values, reynolds_values, mach_values)

        print(f"Processing airfoil: {airfoil_name} (airfoil_num={airfoil_num})")

        normalized_geometry = normalise_geometry(geometry_data)
        flat_coords = []
        for x, y in normalized_geometry:
            flat_coords.extend([x, y])
        
        # Check if the geometry data is already in the input CSV
        for ncrit, re, mach in param_combinations:
            error_key = (airfoil_num, ncrit, mach, re)
            if error_key in existing_errors:
                print(f"  Skipping (exists in error log): Ncrit={ncrit}, Re={re:.1e}, Mach={mach}")
                continue

            expected_alphas = {-5.0, 0.0, 5.0, 10.0, 15.0}
            existing_alphas = set()
            for alpha in expected_alphas:
                entry_key = (airfoil_num, ncrit, re, mach, alpha)
                if entry_key in existing_entries:
                    existing_alphas.add(alpha)

            if existing_alphas == expected_alphas:
                print(f"  Skipping (complete in input): Ncrit={ncrit}, Re={re:.1e}, Mach={mach}")
                continue

            print(f"  Running Ncrit={ncrit}, Re={re:.1e}, Mach={mach}")
            success, result_data = run_xfoil(input_file, ncrit, re, mach)
            ## Check if the run was successful and if data was parsed correctly
            if success:
                parsed_data = result_data.get('parsed_data', [])
                obtained_alphas = {row[0] for row in parsed_data}

                new_entries = 0
                for row in parsed_data:
                    alpha = row[0]
                    entry_key = (airfoil_num, ncrit, re, mach, alpha)
                    if entry_key in existing_entries:
                        continue

                    inp_row = [airfoil_num] + flat_coords + [ncrit, re, mach, alpha]
                    with open(inp_param_csv, 'a', newline='') as f:
                        csv.writer(f).writerow(inp_row)
                    
                    out_row = [airfoil_num, row[1], row[2], row[4], row[3], row[5], row[6]]
                    with open(out_param_csv, 'a', newline='') as f:
                        csv.writer(f).writerow(out_row)
                    
                    existing_entries.add(entry_key)
                    new_entries += 1

                print(f"  Added {new_entries} new entries")

                missing_alphas = sorted(list(expected_alphas - obtained_alphas))
                if missing_alphas:
                    alphalst_str = ' '.join(map(str, missing_alphas))
                    with open(err_param_csv, 'a', newline='') as f:
                        csv.writer(f).writerow([airfoil_num, ncrit, mach, re, alphalst_str])
                    existing_errors.add(error_key)
            else:
                alphalst_str = ' '.join(map(str, sorted(expected_alphas)))
                with open(err_param_csv, 'a', newline='') as f:
                    csv.writer(f).writerow([airfoil_num, ncrit, mach, re, alphalst_str])
                existing_errors.add(error_key)


# Write the airfoil dictionary to the air_leg CSV file
# This ensures that the airfoil numbers are updated in case of new registrations.
with open(air_leg_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["airfoil_num", "airfoil_name"])
    for name, num in sorted(airfoil_dict.items(), key=lambda x: x[1]):
        writer.writerow([num, name])

print("Processing complete. Check the data/xfoil_results directory for output files.")