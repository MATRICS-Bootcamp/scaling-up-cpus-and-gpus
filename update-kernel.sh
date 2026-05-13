#!/bin/bash

echo "Setting up Scaling kernels..."

# Create the first kernel directory
echo "Creating the dask kernel directory..."
mkdir -p ~/.local/share/jupyter/kernels/dask-distribute

# Create the first kernel.json file
echo "Creating dask kernel.json file..."
cat > ~/.local/share/jupyter/kernels/dask-distribute/kernel.json << 'EOF'
{
    "argv": [
        "apptainer", "exec", 
        "/scratch/groups/jfreshwa/containers/dask-distribute.sif",
        "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ],
    "display_name": "Dask Distribute (daskenv)",
    "language": "python"
}
EOF

# Create the second kernel directory
echo "Creating the pytorch kernel directory..."
mkdir -p ~/.local/share/jupyter/kernels/pytorch-ddp

# Create the second kernel.json file
echo "Creating pytorch kernel.json file..."
cat > ~/.local/share/jupyter/kernels/pytorch-ddp/kernel.json << 'EOF'
{
    "argv": [
        "apptainer", "exec", "--nv",
        "/scratch/groups/jfreshwa/containers/pytorch-ddp.sif",
        "python", "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ],
    "display_name": "PyTorch DDP (torchenv)",
    "language": "python"
}
EOF

echo "Kernel setup complete!"
echo "The 'Dask Distribute' and 'PyTorch DDP' kernel should now be available in Jupyter Lab."
echo ""
echo "To verify, check that the directories exists:"
echo "ls -la ~/.local/share/jupyter/kernels/"
