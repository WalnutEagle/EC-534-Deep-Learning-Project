from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

# Test function
def test_x2ct_gan(generator, test_loader, device):
    generator.eval()  # Set model to evaluation mode
    total_psnr, total_ssim = 0.0, 0.0
    count = 0

    with torch.no_grad():
        for pa_xray, lat_xray, ct_volume in test_loader:
            pa_xray, lat_xray, ct_volume = pa_xray.to(device), lat_xray.to(device), ct_volume.to(device)
            
            # Generate CT volume
            fake_ct = generator(pa_xray, lat_xray)
            
            # Compute PSNR and SSIM
            for i in range(fake_ct.shape[0]): 
                fake_ct_np = fake_ct[i].cpu().numpy().squeeze()
                ct_volume_np = ct_volume[i].cpu().numpy().squeeze()
                
                total_psnr += psnr(ct_volume_np, fake_ct_np, data_range=ct_volume_np.max() - ct_volume_np.min())
                total_ssim += ssim(ct_volume_np, fake_ct_np, data_range=ct_volume_np.max() - ct_volume_np.min())
                count += 1

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"Test Results - PSNR: {avg_psnr:.4f}, SSIM: {avg_ssim:.4f}")

test_x2ct_gan(generator, test_loader, device)
