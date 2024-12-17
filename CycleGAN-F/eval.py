def evaluate_model(test_loader, generator_2d_to_3d, device):
    generator_2d_to_3d.eval()

    total_ssim = 0.0
    total_psnr = 0.0
    num_samples = 0

    with torch.no_grad():
        for batch in test_loader:
            frontal = batch['frontal'].to(device)  

            # generating 3D output
            fake_3d = generator_2d_to_3d(frontal)

            # looping over batch
            for i in range(fake_3d.size(0)):
                generated_3d = fake_3d[i].cpu().numpy()
                original = frontal[i].cpu().numpy()
                
                generated_3d = np.clip(generated_3d, 0, 1)  # normalize to [0, 1]
                original = np.clip(original, 0, 1)

                # resizing the original 2D frontal input to match the 3D slice dimensions
                resized_original = np.resize(original, generated_3d.shape[1:])  

                # computing metrics slice-by-slice
                for slice_idx in range(generated_3d.shape[0]):
                    slice_generated = generated_3d[slice_idx]
                    slice_original = resized_original  

                    total_ssim += ssim(slice_generated, slice_original, data_range=1.0, win_size=3)
                    total_psnr += psnr(slice_generated, slice_original, data_range=1.0)
                    num_samples += 1

    # average metrics
    avg_ssim = total_ssim / num_samples if num_samples > 0 else 0
    avg_psnr = total_psnr / num_samples if num_samples > 0 else 0
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average PSNR: {avg_psnr:.4f}")

    return avg_ssim, avg_psnr

# evaluating the model
avg_ssim, avg_psnr = evaluate_model(test_loader, generator_2d_to_3d, device)
