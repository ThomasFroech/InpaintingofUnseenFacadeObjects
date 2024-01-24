import text_prompt_toolkit as tpt
import os
import glob
import time

# Set the path to the folder containing your .png files
folder_path = ""

# Use the glob module to get a list of all .png files in the folder
png_files = glob.glob(os.path.join(folder_path, "*.png"))

# Record the start time
start_time = time.time()

# Iterate over the list of .png files
for png_file in png_files:
    print(png_file)
    test = tpt.build_text_prompt(png_file, config_path="/configAutomaticTextPrompt.json")
    print(test)

# Record the end time
end_time = time.time()

# Calculate and print the runtime
runtime = end_time - start_time
print(f"Total runtime: {runtime} seconds")