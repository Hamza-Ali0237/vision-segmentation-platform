import torch
import numpy as np
import matplotlib
matplotlib.use("Agg") # no display needed, works inside Docker / SageMaker
import matplotlib.pyplot as plt
from pathlib import Path


def save_predictions(model, dataloader, device, output_dir, num_samples=8, threshold=0.5):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    saved_paths = []
    count = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).float()

            for i in range(images.size(0)):
                if count >= num_samples:
                    break

                img = images[i].cpu().permute(1, 2, 0).numpy()
                mask = masks[i].cpu().squeeze().numpy()
                pred = preds[i].cpu().squeeze().numpy()

                # Undo ImageNet normalisation for display
                mean = np.array([0.485, 0.456, 0.406])
                std  = np.array([0.229, 0.224, 0.225])
                img  = np.clip(img * std + mean, 0, 1)

                fig, axes = plt.subplots(1, 3, figsize=(12, 4))
                axes[0].imshow(img);axes[0].set_title("Image"); axes[0].axis("off")
                axes[1].imshow(mask, cmap="gray"); axes[1].set_title("Ground Truth"); axes[1].axis("off")
                axes[2].imshow(pred, cmap="gray"); axes[2].set_title("Prediction"); axes[2].axis("off")

                out_path = output_dir / f"sample_{count:04d}.png"
                plt.tight_layout()
                plt.savefig(out_path, dpi=100, bbox_inches="tight")
                plt.close(fig)

                saved_paths.append(str(out_path))
                count += 1

            if count >= num_samples:
                break

    return saved_paths