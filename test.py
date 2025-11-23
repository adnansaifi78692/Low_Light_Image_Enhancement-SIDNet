import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import pyiqa  # NIQE
from piq import LPIPS  # LPIPS
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import numpy as np
import csv


def make_dirs(path):
    os.makedirs(path, exist_ok=True)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
class Config:
    device       = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    ckpt_path    = r'C:\Users\ADNAN SAIFI\Desktop\New folder\ckpt_color3\sid_epoch_100.pth'
    input_dir    = r'C:\Users\ADNAN SAIFI\Desktop\Test'
    normal_dir   = r'C:\Users\ADNAN SAIFI\Desktop\LOLv2\Real_captured\Test\Normal'
    output_dir   = 'enhanced_images_synthetic'
    img_exts     = ('.png', '.jpg', '.jpeg', '.bmp')
    csv_file     = 'metrics.csv'

make_dirs(Config.output_dir)

# --------------------------------------------------
# MODEL DEFINITION
# --------------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
        )
    def forward(self, x):
        return x + self.net(x)

class SIDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in   = nn.Conv2d(3, 32, 3, padding=1, bias=False)
        self.down1     = nn.Conv2d(32, 64, 4, stride=2, padding=1, bias=False)
        self.rb1       = ResidualBlock(64)
        self.down2     = nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)
        self.rb2       = ResidualBlock(128)
        self.rb_mid    = ResidualBlock(128)
        self.rb_mid2   = ResidualBlock(128)
        self.up1       = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False)
        self.rb3       = ResidualBlock(64)
        self.up2       = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False)
        self.rb4       = ResidualBlock(32)
        self.conv_out  = nn.Conv2d(32, 3, 3, padding=1)

    def forward(self, x):
        x0 = self.conv_in(x)
        x1 = F.relu(self.rb1(self.down1(x0)))
        x2 = F.relu(self.rb2(self.down2(x1)))
        xm = self.rb_mid(self.rb_mid2(x2))
        y1 = F.relu(self.rb3(self.up1(xm))) + x1
        y2 = F.relu(self.rb4(self.up2(y1))) + x0
        return torch.sigmoid(self.conv_out(y2))

# --------------------------------------------------
# INFERENCE FUNCTION
# --------------------------------------------------
def enhance_image(image_path, ground_truth_path, model, device, transform, save_dir, niqe_metric, lpips_metric):
    try:
        # Load images
        low_img = Image.open(image_path).convert('RGB')
        norm_img = Image.open(ground_truth_path).convert('RGB')
        
        # Transform
        input_tensor = transform(low_img).unsqueeze(0).to(device)
        gt_tensor = transform(norm_img).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            out_tensor = model(input_tensor)
            out_tensor = out_tensor.clamp(0, 1)

        # Convert for metric calculation
        out_np = out_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
        gt_np = gt_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

        # PSNR & SSIM
        psnr = compare_psnr(gt_np, out_np, data_range=1.0)
        ssim = compare_ssim(gt_np, out_np, data_range=1.0, channel_axis=-1, win_size=7)

        # NIQE
        niqe_score = niqe_metric(out_tensor).item()

        # LPIPS
        lpips_score = lpips_metric(out_tensor, gt_tensor).item()

        # Print Metrics
        print(f"Metrics for {os.path.basename(image_path)}:")
        print(f"PSNR:  {psnr:.2f}")
        print(f"SSIM:  {ssim:.4f}")
        print(f"NIQE:  {niqe_score:.4f}")
        print(f"LPIPS: {lpips_score:.4f}")
        print('-' * 50)

        # Save the image
                # Save the image with metrics as a publication-quality plot
            # Save the image with metrics as a clean side-by-side figure
        save_path = os.path.join(save_dir, f"enhanced_{os.path.basename(image_path)}.png")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.subplots_adjust(top=0.85)
        fig.suptitle(f"Enhancement Result: {os.path.basename(image_path)}", fontsize=18, fontweight='bold')

        images = [input_tensor, out_tensor]
        titles = ['Low-Light Input', 'Enhanced Output']
        
        for ax, img_tensor, title in zip(axes, images, titles):
            img_np = img_tensor.squeeze().cpu().permute(1, 2, 0).numpy()
            ax.imshow(np.clip(img_np, 0, 1))
            ax.set_title(title, fontsize=14, weight='bold')
            ax.axis('off')

        # Add metrics as a floating textbox on the Enhanced Output
        metric_text = f"PSNR:  {psnr:.2f} dB\nSSIM:  {ssim:.4f}\nNIQE:  {niqe_score:.4f}\nLPIPS: {lpips_score:.4f}"
        axes[1].text(
            1.05, 0.5, metric_text,
            fontsize=12, ha='left', va='center', transform=axes[1].transAxes,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
        )

        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show(fig)

        print(f"Saved plot to: {save_path}")

        

        
        return [psnr, ssim, niqe_score, lpips_score]

    except Exception as e:
        print(f"Error processing image: {e}")
        return [None, None, None, None]

# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == '__main__':
    model = SIDNet().to(Config.device)
    checkpoint = torch.load(Config.ckpt_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])

    # Metrics
    niqe_metric = pyiqa.create_metric('niqe').to(Config.device)
    lpips_metric = LPIPS(replace_pooling=True).to(Config.device)

    # Create a CSV file to store metrics
    with open(Config.csv_file, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'PSNR', 'SSIM', 'NIQE', 'LPIPS']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Loop through all images in the input directory
        for image_name in tqdm(os.listdir(Config.input_dir)):
            if image_name.lower().endswith(Config.img_exts):
                img_path = os.path.join(Config.input_dir, image_name)
                gt_path = os.path.join(Config.normal_dir, image_name)

                # Process the image and get metrics
                psnr, ssim, niqe, lpips = enhance_image(img_path, gt_path, model, Config.device, transform, Config.output_dir, niqe_metric, lpips_metric)

                if psnr is not None:  # If metrics are valid, save them
                    writer.writerow({
                        'Image': image_name,
                        'PSNR': psnr,
                        'SSIM': ssim,
                        'NIQE': niqe,
                        'LPIPS': lpips
                    })

    print("\nAll evaluations complete.")
