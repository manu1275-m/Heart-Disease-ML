import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

print("\n=== MODEL TYPE ===")
print(type(model))

# If it's a Pipeline
try:
    print("\nPipeline steps:")
    print(model.named_steps)
except Exception:
    print("\nNot a pipeline")

# Try to get feature names
try:
    pre = model.named_steps.get("preprocessor")
    if pre:
        print("\nPreprocessor transformers:")
        print(pre.transformers_)
        print("\nColumns used:", pre.transformers_[0][2])
    else:
        print("\nNo preprocessor in pipeline")
except Exception as e:
    print("\nCould not extract preprocessor:", e)

# Try to get raw model
try:
    if hasattr(model, "named_steps"):
        core = model.named_steps.get("model")
    else:
        core = model

    print("\n=== Core model type ===")
    print(type(core))
except:
    print("\nCould not extract core model")