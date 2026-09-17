from soloWork.dataset_gait import GaitDataset

dataset = GaitDataset("gait_sequences_41_BULG_2.npz")
print("num_classes:", len(dataset.label2idx))
print("len(dataset):", len(dataset))