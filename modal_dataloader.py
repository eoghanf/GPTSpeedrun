#!/usr/bin/env python3
"""
Local script to download files from Huggingface and upload them to Modal volume.
This script runs locally and uses Modal CLI to upload files to avoid size limitations.
"""

import os
import sys
import subprocess
import tempfile
import hashlib
import struct
import argparse
from tqdm import tqdm
from huggingface_hub import hf_hub_download


def run_command(cmd, description, show_progress=False):
    """Run a command and handle errors"""
    if show_progress:
        print(f"{description}...")
    try:
        if show_progress:
            # Show progress for long operations
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with tqdm(desc=description, unit="B", unit_scale=True, leave=False) as pbar:
                while process.poll() is None:
                    pbar.update(1024)  # Arbitrary update
                stdout, stderr = process.communicate()
        else:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            stdout, stderr = result.stdout, result.stderr
        
        if process.returncode != 0 if show_progress else False:
            raise subprocess.CalledProcessError(process.returncode if show_progress else 0, cmd, stdout, stderr)
        
        if show_progress:
            print(f"✓ {description} completed")
        return stdout
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return None


def calculate_file_hash(file_path):
    """Calculate SHA256 hash of a file with progress bar"""
    sha256_hash = hashlib.sha256()
    file_size = os.path.getsize(file_path)
    
    with open(file_path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Calculating hash", leave=False) as pbar:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
                pbar.update(len(chunk))
    
    return sha256_hash.hexdigest()


def preview_binary_file(file_path, num_tokens=10):
    """Preview first few tokens from a binary file (assuming uint16 tokens)"""
    try:
        with open(file_path, 'rb') as f:
            # Read first num_tokens * 2 bytes (assuming uint16 tokens)
            data = f.read(num_tokens * 2)
            if len(data) < num_tokens * 2:
                num_tokens = len(data) // 2
            
            # Unpack as uint16 tokens
            tokens = struct.unpack(f'<{num_tokens}H', data[:num_tokens * 2])
            print(f"  First {num_tokens} tokens: {tokens}")
            
            # Show file size
            f.seek(0, 2)  # Seek to end
            size_mb = f.tell() / (1024 * 1024)
            print(f"  File size: {size_mb:.1f} MB")
            
    except Exception as e:
        print(f"  Could not preview file: {e}")


def verify_modal_upload(volume_name, remote_path, local_hash, temp_dir):
    """Download file from Modal and verify hash matches"""
    try:
        # Download from Modal volume to temp location
        verify_path = os.path.join(temp_dir, f"verify_{os.path.basename(remote_path)}")
        download_cmd = f"modal volume get {volume_name} {remote_path} {verify_path}"
        
        if run_command(download_cmd, f"Downloading {remote_path} for verification"):
            # Calculate hash of downloaded file
            modal_hash = calculate_file_hash(verify_path)
            
            # Clean up verification file
            os.remove(verify_path)
            
            if local_hash == modal_hash:
                print(f"  ✓ Hash verification passed: {local_hash[:16]}...")
                return True
            else:
                print(f"  ✗ Hash mismatch! Local: {local_hash[:16]}..., Modal: {modal_hash[:16]}...")
                return False
        else:
            print(f"  ✗ Could not download file for verification")
            return False
            
    except Exception as e:
        print(f"  ✗ Verification failed: {e}")
        return False


def download_and_upload_file(filename, temp_dir, volume_name, skip_verification=False):
    """Download a file from HF and upload to Modal volume"""
    print(f"\n--- Processing {filename} ---")
    
    # Download file to temp directory
    try:
        local_path = hf_hub_download(
            repo_id="kjj0/finewebedu10B-gpt2",
            filename=filename,
            repo_type="dataset",
            local_dir=temp_dir
        )
        print(f"✓ Downloaded {filename} to {local_path}")
        
        # Preview file content
        print("📋 File preview:")
        preview_binary_file(local_path)
        
        # Calculate hash before upload (only if verification enabled)
        local_hash = None
        if not skip_verification:
            local_hash = calculate_file_hash(local_path)
            print(f"📊 Local file hash: {local_hash[:16]}...")
        
        # Upload to Modal volume using CLI
        upload_cmd = f"modal volume put {volume_name} {local_path} /{filename}"
        if run_command(upload_cmd, f"Uploading {filename} to Modal volume", show_progress=True):
            if skip_verification:
                # Skip verification, just clean up
                os.remove(local_path)
                print(f"✓ Cleaned up local file {filename} (verification skipped)")
                return True
            else:
                # Verify upload by downloading and comparing hash
                print("🔍 Verifying upload...")
                if verify_modal_upload(volume_name, f"/{filename}", local_hash, temp_dir):
                    # Clean up local file only if verification passes
                    os.remove(local_path)
                    print(f"✓ Cleaned up local file {filename}")
                    return True
                else:
                    print(f"✗ Upload verification failed for {filename}")
                    return False
        else:
            print(f"✗ Failed to upload {filename}")
            return False
            
    except Exception as e:
        print(f"✗ Error processing {filename}: {e}")
        return False


def main():
    """Main function to download and upload all files"""
    parser = argparse.ArgumentParser(description="Download files from HuggingFace and upload to Modal volume")
    parser.add_argument("--skip-verification", action="store_true", 
                       help="Skip hash verification (faster but less safe)")
    parser.add_argument("--volume", default="fineweb-volume", 
                       help="Modal volume name (default: fineweb-volume)")
    parser.add_argument("--chunks", type=int, default=8,
                       help="Number of training chunks to process (default: 8)")
    
    args = parser.parse_args()
    
    volume_name = args.volume
    num_chunks = args.chunks
    skip_verification = args.skip_verification
    
    if skip_verification:
        print("⚠️  Hash verification is DISABLED - uploads will not be verified")
    
    # Check if Modal CLI is available
    if not run_command("modal --version", "Checking Modal CLI"):
        print("Error: Modal CLI not found. Please install it with: pip install modal")
        sys.exit(1)
    
    # Create temporary directory for downloads
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Process validation file first
        print("\n=== Processing validation file ===")
        if not download_and_upload_file("finewebedu_val_000000.bin", temp_dir, volume_name, skip_verification):
            print("Failed to process validation file")
            sys.exit(1)
        
        # Process training files with overall progress
        print("\n=== Processing training files ===")
        successful_uploads = 0
        failed_uploads = 0
        
        with tqdm(total=num_chunks, desc="Overall progress", unit="files") as overall_pbar:
            for i in range(1, num_chunks + 1):
                filename = f"finewebedu_train_{i:06d}.bin"
                if download_and_upload_file(filename, temp_dir, volume_name, skip_verification):
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                
                overall_pbar.update(1)
                overall_pbar.set_postfix(success=successful_uploads, failed=failed_uploads)
    
    print(f"\n=== Summary ===")
    print(f"Successfully uploaded: {successful_uploads + 1} files")  # +1 for validation
    print(f"Failed uploads: {failed_uploads}")
    
    if failed_uploads > 0:
        print(f"⚠️  {failed_uploads} files failed to upload")
        sys.exit(1)
    else:
        print("✅ All files uploaded successfully!")


if __name__ == "__main__":
    main()