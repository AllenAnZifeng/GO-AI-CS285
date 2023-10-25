import pickle
import matplotlib.pyplot as plt

# Load data with pickle
with open('training_logs6x6.pkl', 'rb') as f:
    data = pickle.load(f)

# Parse the data
parsed_data = {
    "ITR": [],
    "TIME": [],
    "REPLAY": [],
    "C_ACC": [],
    "C_LOSS": [],
    "A_ACC": [],
    "A_LOSS": [],
    "G_LOSS": [],
    "C_WR": [],
    "R_WR": [],
    "G_WR": []
}

for line in data:
    fields = line.split('\t')
    parsed_data["TIME"].append(fields[0])
    parsed_data["ITR"].append(int(fields[1]))
    parsed_data["REPLAY"].append(int(fields[2]))
    parsed_data["C_ACC"].append(float(fields[3]))
    parsed_data["C_LOSS"].append(float(fields[4]))
    parsed_data["A_ACC"].append(float(fields[5]))
    parsed_data["A_LOSS"].append(float(fields[6]))
    parsed_data["G_LOSS"].append(float(fields[7].replace("____", "0.0")))
    parsed_data["C_WR"].append(float(fields[8]))
    parsed_data["R_WR"].append(float(fields[9]))
    parsed_data["G_WR"].append(float(fields[10]))

# Plotting
keys_to_plot = ["C_ACC", "C_LOSS", "A_ACC", "A_LOSS", "G_LOSS", "C_WR", "R_WR", "G_WR"]
num_rows = (len(keys_to_plot) + 1) // 2

fig, axs = plt.subplots(num_rows, 2, figsize=(14, 5 * num_rows))

for i, key in enumerate(keys_to_plot):
    ax = axs[i // 2, i % 2]
    ax.plot(parsed_data["ITR"], parsed_data[key], label=key, color='b')
    ax.set_title(key)
    ax.set_xlabel('ITR')
    ax.set_ylabel(key)
    ax.grid(True)

# If there's an odd number of plots, hide the last unused subplot
if len(keys_to_plot) % 2:
    axs[-1, -1].axis('off')

plt.tight_layout()
plt.savefig('output_graphs.png')
plt.show()

