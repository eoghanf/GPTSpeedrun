import modal, os, subprocess

app = modal.App('KJ-speedrun')

training_image = (
    modal.Image.debian_slim(python_version="3.13")
    .apt_install('git')
    .run_commands(f"git clone https://github.com/KellerJordan/modded-nanogpt.git && cd modded-nanogpt &&"
    f"pip install -r requirements.txt &&"
    f"pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --upgrade &&"
    f"python data/cached_fineweb10B.py 8")
)


@app.function(image=training_image, gpu='H100', timeout=5000)
def train_model():
    os.chdir('/modded-nanogpt')  # Change to the repo directory

    # Run torchrun with single GPU
    result = subprocess.run([
        'torchrun',
        '--standalone',
        '--nproc_per_node=1',  # Changed to 1 for single GPU
        'train_gpt.py'
    ], capture_output=True, text=True)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    print("Return code:", result.returncode)

    return result.returncode == 0

