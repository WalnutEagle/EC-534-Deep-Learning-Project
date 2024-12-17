# EVALUATION LOOP

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def evaluate_model(test_loader, generator_2d_to_3d, generator_3d_to_2d, device):
    generator_2d_to_3d.eval()
    generator_3d_to_2d.eval()

    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            frontal, lateral = batch['frontal'].to(device), batch['lateral'].to(device)
            input_xrays = torch.cat((frontal, lateral), dim=1)  # combined input with 2 channels

            # generating 3D and reconstructing 2D (cycle)
            fake_3d = generator_2d_to_3d(input_xrays)
            reconstructed_xrays = generator_3d_to_2d(fake_3d)

            # looping over batch
            for i in range(reconstructed_xrays.size(0)):
                reconstructed = reconstructed_xrays[i].cpu().numpy()
                original = input_xrays[i].cpu().numpy()

                # processing channels separately
                for c in range(reconstructed.shape[0]):  # Loop over 2 channels
                    reconstructed_channel = np.clip(reconstructed[c], 0, 1)
                    original_channel = np.clip(original[c], 0, 1)

                    # computing metrics for the channel
                    total_ssim += ssim(reconstructed_channel, original_channel, data_range=1.0, win_size=3)
                    total_psnr += psnr(reconstructed_channel, original_channel, data_range=1.0)
                    num_samples += 1

    # average 
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

    return avg_ssim, avg_psnr

# evaluating the model
avg_ssim, avg_psnr = evaluate_model(test_loader, generator_2d_to_3d, generator_3d_to_2d, device)
