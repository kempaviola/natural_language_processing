from datasets import load_dataset

ds = load_dataset("google/civil_comments")

print(ds)
print()
print("Features:", ds["train"].features)
print()
print("Example:", ds["train"][0])
