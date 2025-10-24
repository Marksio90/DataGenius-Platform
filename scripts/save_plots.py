
import os, json
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt

def main():
    os.makedirs("artifacts/plots", exist_ok=True)
    preds_path = "artifacts/models/preds.json"
    if not os.path.exists(preds_path):
        print("No preds.json; skipping plots.")
        return
    data = json.load(open(preds_path,"r",encoding="utf-8"))
    y = np.array(data.get("y_true", []))
    p = np.array(data.get("y_prob", []))
    if y.size==0 or p.size==0 or len(set(y.tolist()))<2:
        print("Not enough data for plots.")
        return

    fpr, tpr, _ = roc_curve(y, p)
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1]); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.savefig("artifacts/plots/roc.png", dpi=120); plt.close()
    prec, rec, _ = precision_recall_curve(y, p)
    plt.figure(); plt.plot(rec,prec); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR"); plt.savefig("artifacts/plots/pr.png", dpi=120); plt.close()
    yhat = (p>=0.5).astype(int)
    cm = confusion_matrix(y, yhat)
    plt.figure(); plt.imshow(cm); plt.title("Confusion"); plt.colorbar(); plt.savefig("artifacts/plots/cm.png", dpi=120); plt.close()

if __name__ == "__main__":
    main()
