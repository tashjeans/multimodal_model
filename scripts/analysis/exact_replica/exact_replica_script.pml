
# EXACT replica of the working PyMOL script from the notebook
# This should give you the same alignment and appearance

# ————— Load Structures (EXACT order from working script) —————
load /home/natasha/multimodal_model/7ow5.cif, native
load /home/natasha/multimodal_model/outputs/boltz_out/boltz_results_70W5_with_MSA/predictions/70W5/70W5_model_0.cif, test

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
png exact_replica/exact_replica_aligned.png, width=800, height=600, ray=1

# Also create a high-res version with white background
png exact_replica/exact_replica_highres.png, width=1600, height=1200, ray=1, dpi=300

# Save session file
save exact_replica/exact_replica_session.pse

# Quit PyMOL
quit
