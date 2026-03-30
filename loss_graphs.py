import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("loss_log.csv")
components = ["loss", "coordinate", "dimension", "object_confidence", "noobject_confidence", "class_confidence"]

for comp in components:
    plt.figure()
    plt.plot(data["epoch"], data[f"train_{comp}"], label="train")
    plt.plot(data["epoch"], data[f"validation_{comp}"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel(f"{comp} loss")
    plt.title(f"{comp} Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{comp}_loss.png")  # optional: save figure
    plt.show()