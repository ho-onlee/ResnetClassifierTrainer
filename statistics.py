from src.stats import datasetStats

# # Load dataset
# try:
#     torch_dataset = load_torch_dataset()
# except ValueError as e:
#     print(e)
#     print("No extracted features found. Running process_data() to generate features...")
#     process_data()
#     torch_dataset = load_torch_dataset()


datasetStats()
