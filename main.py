import os
import time
import numpy as np
import torch
from config import NUM_RUNS, EPOCHS, CLASS_INFO, NUM_CLASSES
from train import setup_training, train_model
from evaluate import evaluate, generate_classification_map, save_metrics
from dataset import HyperspectralDataset
from model import GeoFusionNet

def main():
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('results', exist_ok=True)
    
    # Get number of bands from data
    train_hsi = np.load('data/train_hsi.npy')
    num_bands = train_hsi.shape[-1]
    
    # Initialize metrics storage
    all_metrics = []
    class_names = {i+1: info[1] for i, info in enumerate(CLASS_INFO)}
    
    for run in range(NUM_RUNS):
        print(f"\n=== Starting Run {run+1}/{NUM_RUNS} ===")
        run_dir = f'results/run_{run+1}'
        os.makedirs(run_dir, exist_ok=True)
        
        # Setup training
        model, optimizer, scheduler, train_loader, test_loader, weights = setup_training(
            run_dir, num_bands, NUM_CLASSES, device)
        
        best_val_oa = 0
        train_losses = []
        val_oa_history = []
        
        for epoch in range(EPOCHS):
            # Train
            epoch_loss = train_model(
                model, train_loader, optimizer, device, epoch, weights)
            train_losses.append(epoch_loss)
            
            # Validate
            val_oa, val_aa, val_kappa, _, _ = evaluate(model, test_loader, device)
            val_oa_history.append(val_oa)
            
            print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Val OA={val_oa:.4f}")
            
            # Update scheduler and save best model
            scheduler.step(val_oa)
            if val_oa > best_val_oa:
                best_val_oa = val_oa
                torch.save(model.state_dict(), f'{run_dir}/best_model.pth')
                print(f"Saved new best model at epoch {epoch+1} with OA: {val_oa:.4f}")
        
        # Final test evaluation
        model.load_state_dict(torch.load(f'{run_dir}/best_model.pth'))
        test_oa, test_aa, test_kappa, test_cm, test_class_acc = evaluate(
            model, test_loader, device)
        all_metrics.append((test_oa, test_aa, test_kappa, test_cm, test_class_acc))
        
        print(f"\nRun {run+1} Test Results:")
        print(f"OA: {test_oa:.4f}, AA: {test_aa:.4f}, Kappa: {test_kappa:.4f}")
        
        # Generate classification map
        generate_classification_map(model, device, run_dir)
    
    # Save all metrics
    save_metrics(all_metrics, class_names, 'results')
    
    print(f"\nTotal execution time: {(time.time()-start_time)/60:.2f} minutes")
    print("Results saved in 'results' directory")

if __name__ == "__main__":
    main()