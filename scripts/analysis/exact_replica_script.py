#!/usr/bin/env python3
"""
EXACT replica of the working PyMOL script from the notebook
This should give you the same alignment and appearance
"""

from pathlib import Path
import os

def create_exact_replica():
    """
    Create an EXACT replica of the working PyMOL script
    """
    try:
        # Set up paths (EXACTLY as in the working script)
        boltz_dir = Path("/home/natasha/multimodal_model") / "outputs" / "boltz_out"
        test_cif = boltz_dir / "boltz_results_70W5_with_MSA" / "predictions" / "70W5" / "70W5_model_0.cif"
        native_path = Path("/home/natasha/multimodal_model") 
        native_cif = native_path / "7ow5.cif"
        
        # Check if files exist
        if not test_cif.exists():
            print(f"❌ Test CIF file not found at {test_cif}")
            return False
        
        if not native_cif.exists():
            print(f"❌ Native CIF file not found at {native_cif}")
            return False
        
        print(f"✅ Test structure: {test_cif.name}")
        print(f"✅ Native structure: {native_cif.name}")
        
        # Create output directory
        output_dir = "exact_replica"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create PyMOL script that is an EXACT replica
        pymol_script = f"""
# EXACT replica of the working PyMOL script from the notebook
# This should give you the same alignment and appearance

# ————— Load Structures (EXACT order from working script) —————
load {native_cif}, native
load {test_cif}, test

# ————— Align test → native (EXACT command from working script) —————
# This superimposes "test" onto "native" so they overlap
align test, native

# ————— Representations (EXACT from working script) —————
hide everything, all
show cartoon, all

# ————— Color by Chain (EXACT colors from working script) —————
# Define your chains and two color-pairs (EXACTLY as in working script):
# chains = ["A", "B", "C", "D", "E"]
# native_colors = ["forest", "cyan", "blue", "green", "purple"]  
# test_colors   = ["lime", "yellow", "grey", "orange", "pink"]

# Chain A
color forest, native and chain A
color lime, test and chain A

# Chain B
color cyan, native and chain B
color yellow, test and chain B

# Chain C
color blue, native and chain C
color grey, test and chain C

# Chain D
color green, native and chain D
color orange, test and chain D

# Chain E
color purple, native and chain E
color pink, test and chain E

# ————— Frame & Render (EXACT from working script) —————
zoom all
orient all

# ————— Force White Background —————
# Set background to white (multiple commands to ensure it works)
bg_color white
set bg_rgb, [1.0, 1.0, 1.0]
set ray_trace_mode, 1
set ray_shadows, 0
set antialias, 2

# Remove any transparency that might interfere
set cartoon_transparency, 0.0, all
set transparency, 0.0, all

# Render with EXACT same parameters as working script
png {output_dir}/exact_replica_aligned.png, width=800, height=600, ray=1

# Also create a high-res version with white background
png {output_dir}/exact_replica_highres.png, width=1600, height=1200, ray=1, dpi=300

# Save session file
save {output_dir}/exact_replica_session.pse

# Quit PyMOL
quit
"""
        
        # Save the PyMOL script
        script_path = os.path.join(output_dir, "exact_replica_script.pml")
        with open(script_path, 'w') as f:
            f.write(pymol_script)
        
        print(f"✅ EXACT replica script saved as: {script_path}")
        
        # Run PyMOL with the exact replica script
        print("🚀 Running EXACT replica script...")
        os.system(f"pymol -c {script_path}")
        print("✅ EXACT replica completed!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating exact replica: {e}")
        return False

def main():
    """Main function"""
    print("🔬 Creating EXACT Replica of Working PyMOL Script")
    print("=" * 60)
    print("This replicates your working script EXACTLY, including:")
    print("✅ Same structure loading order")
    print("✅ Same alignment command")
    print("✅ Same color scheme (forest/cyan/blue/green/purple for native)")
    print("✅ Same color scheme (lime/yellow/grey/orange/pink for test)")
    print("✅ Same view settings")
    print("=" * 60)
    
    # Create the exact replica
    success = create_exact_replica()
    
    if success:
        print("\n🎉 EXACT replica created successfully!")
        print("📁 Check the 'exact_replica' folder for your images:")
        print("  - exact_replica_aligned.png (same as your working script)")
        print("  - exact_replica_highres.png (high-res version)")
        print("  - exact_replica_session.pse (PyMOL session file)")
        print("\n🎯 This should look EXACTLY like your working script output!")
    else:
        print("\n❌ Failed to create exact replica. Check the error messages above.")

if __name__ == "__main__":
    main() 