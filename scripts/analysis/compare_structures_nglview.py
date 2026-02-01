#!/usr/bin/env python3
"""
Compare Boltz prediction vs PDB native structures side by side
Using NGLView for clean visualization without watermarks
"""

import nglview as nv
from pathlib import Path
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

def create_comparison_view(boltz_cif, native_cif, output_dir="structure_comparison"):
    """
    Create a side-by-side comparison of Boltz vs PDB structures
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert paths to strings
        boltz_str = str(boltz_cif)
        native_str = str(native_cif)
        
        print(f"🔬 Loading Boltz structure: {boltz_cif.name}")
        print(f"📊 Loading native PDB structure: {native_cif.name}")
        
        # Create two separate views
        boltz_view = nv.show_structure_file(boltz_str)
        native_view = nv.show_structure_file(native_str)
        
        # Configure Boltz view (left side)
        boltz_view.clear_representations()
        boltz_view.add_cartoon(color='lime')  # Bright green for Boltz
        boltz_view.add_ball_and_stick(color='lime')
        boltz_view.camera = 'orthographic'
        boltz_view.layout.width = '400px'
        boltz_view.layout.height = '500px'
        
        # Configure native view (right side)
        native_view.clear_representations()
        native_view.add_cartoon(color='cyan')  # Blue for native PDB
        native_view.add_ball_and_stick(color='cyan')
        native_view.camera = 'orthographic'
        native_view.layout.width = '400px'
        native_view.layout.height = '500px'
        
        # Create HTML comparison page
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Boltz vs PDB Structure Comparison</title>
    <script src="https://unpkg.com/ngl@0.10.4/dist/ngl.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            text-align: center;
            margin-bottom: 20px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
        }}
        .comparison-container {{
            display: flex;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .structure-panel {{
            flex: 1;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .structure-title {{
            text-align: center;
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }}
        .boltz-title {{
            background-color: #90EE90;
            color: #006400;
        }}
        .native-title {{
            background-color: #87CEEB;
            color: #000080;
        }}
        .viewport {{
            width: 100%;
            height: 400px;
            border: 2px solid #ddd;
            border-radius: 5px;
        }}
        .legend {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 10px 0;
        }}
        .color-box {{
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 3px;
        }}
        .controls {{
            text-align: center;
            margin: 20px 0;
        }}
        .control-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            margin: 5px;
            border-radius: 5px;
            cursor: pointer;
        }}
        .control-btn:hover {{
            background: #5a6fd8;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Molecular Structure Comparison</h1>
        <h2>Boltz Prediction vs Native PDB Structure</h2>
        <p>Interactive 3D visualization with color-coded chains</p>
    </div>
    
    <div class="comparison-container">
        <div class="structure-panel">
            <div class="structure-title boltz-title">🎯 Boltz Prediction Structure</div>
            <div id="boltz-viewport" class="viewport"></div>
        </div>
        
        <div class="structure-panel">
            <div class="structure-title native-title">📊 Native PDB Structure</div>
            <div id="native-viewport" class="viewport"></div>
        </div>
    </div>
    
    <div class="controls">
        <button class="control-btn" onclick="resetViews()">🔄 Reset Views</button>
        <button class="control-btn" onclick="alignStructures()">⚡ Align Structures</button>
        <button class="control-btn" onclick="toggleRepresentations()">🎨 Toggle Representations</button>
    </div>
    
    <div class="legend">
        <h3>🎨 Color Legend</h3>
        <div class="legend-item">
            <div class="color-box" style="background-color: #90EE90;"></div>
            <span><strong>Boltz Prediction:</strong> Bright green - AI-generated structure</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: #87CEEB;"></div>
            <span><strong>Native PDB:</strong> Blue - experimentally determined structure</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: #FFD700;"></div>
            <span><strong>Chain A:</strong> Gold - typically the main chain</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: #FF69B4;"></div>
            <span><strong>Chain B:</strong> Hot pink - secondary chain</span>
        </div>
        <div class="legend-item">
            <div class="color-box" style="background-color: #32CD32;"></div>
            <span><strong>Chain C:</strong> Lime green - tertiary chain</span>
        </div>
    </div>

    <script>
        // Initialize NGL stages
        var boltzStage = new NGL.Stage("boltz-viewport");
        var nativeStage = new NGL.Stage("native-viewport");
        
        // Load structures
        boltzStage.loadFile("{boltz_str}").then(function(boltzComponent) {{
            boltzComponent.addRepresentation("cartoon", {{color: "lime"}});
            boltzComponent.addRepresentation("ball+stick", {{color: "lime"}});
            
            // Color by chain
            var chains = ["A", "B", "C", "D", "E"];
            var colors = ["gold", "hotpink", "limegreen", "orange", "purple"];
            
            chains.forEach(function(chain, index) {{
                var selection = ":" + chain;
                boltzComponent.addRepresentation("cartoon", {{
                    color: colors[index % colors.length],
                    sele: selection
                }});
            }});
            
            boltzStage.autoView();
        }});
        
        nativeStage.loadFile("{native_str}").then(function(nativeComponent) {{
            nativeComponent.addRepresentation("cartoon", {{color: "cyan"}});
            nativeComponent.addRepresentation("ball+stick", {{color: "cyan"}});
            
            // Color by chain
            var chains = ["A", "B", "C", "D", "E"];
            var colors = ["gold", "hotpink", "limegreen", "orange", "purple"];
            
            chains.forEach(function(chain, index) {{
                var selection = ":" + chain;
                nativeComponent.addRepresentation("cartoon", {{
                    color: colors[index % colors.length],
                    sele: selection
                }});
            }});
            
            nativeStage.autoView();
        }});
        
        // Control functions
        function resetViews() {{
            boltzStage.autoView();
            nativeStage.autoView();
        }}
        
        function alignStructures() {{
            // This would require more complex logic for actual alignment
            alert("Alignment feature would require additional implementation");
        }}
        
        function toggleRepresentations() {{
            // Toggle between cartoon and ball+stick
            var boltzReps = boltzStage.getRepresentations();
            var nativeReps = nativeStage.getRepresentations();
            
            boltzReps.forEach(function(rep) {{
                rep.setVisible(!rep.visible);
            }});
            nativeReps.forEach(function(rep) {{
                rep.setVisible(!rep.visible);
            }});
        }}
    </script>
</body>
</html>
        """
        
        # Save the HTML file
        html_path = os.path.join(output_dir, "boltz_vs_pdb_comparison.html")
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"✅ Comparison HTML saved as: {html_path}")
        
        # Also create a simple side-by-side view using NGLView widgets
        try:
            # Create a combined view with both structures
            combined_view = nv.show_structure_file(boltz_str)
            combined_view.add_structure(native_str)
            
            # Configure the combined view
            combined_view.clear_representations()
            combined_view.add_cartoon(color='lime', sele='0')  # Boltz
            combined_view.add_cartoon(color='cyan', sele='1')  # Native
            combined_view.add_ball_and_stick(color='lime', sele='0')
            combined_view.add_ball_and_stick(color='cyan', sele='1')
            
            # Save as HTML
            combined_html_path = os.path.join(output_dir, "combined_view.html")
            combined_view.save_html(combined_html_path)
            print(f"✅ Combined view HTML saved as: {combined_html_path}")
            
        except Exception as e:
            print(f"⚠️  Combined view creation failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating comparison: {e}")
        return False

def main():
    """Main function to create structure comparison"""
    print("🔬 Boltz vs PDB Structure Comparison Tool")
    print("=" * 50)
    
    # Set up paths
    boltz_dir = Path("/home/natasha/multimodal_model") / "outputs" / "boltz_out"
    boltz_cif = boltz_dir / "boltz_results_70W5_with_MSA" / "predictions" / "70W5" / "70W5_model_0.cif"
    native_cif = Path("/home/natasha/multimodal_model") / "7ow5.cif"
    
    # Check if files exist
    if not boltz_cif.exists():
        print(f"❌ Boltz CIF file not found at {boltz_cif}")
        return
    
    if not native_cif.exists():
        print(f"❌ Native PDB CIF file not found at {native_cif}")
        return
    
    print(f"✅ Boltz structure: {boltz_cif.name}")
    print(f"✅ Native structure: {native_cif.name}")
    
    # Create the comparison
    success = create_comparison_view(boltz_cif, native_cif)
    
    if success:
        print("\n🎉 Comparison created successfully!")
        print("📱 Open 'boltz_vs_pdb_comparison.html' in your web browser to view")
        print("🔍 Features:")
        print("  - Side-by-side comparison")
        print("  - Color-coded chains (A=Gold, B=Pink, C=Lime, etc.)")
        print("  - Interactive 3D controls")
        print("  - No watermarks!")
    else:
        print("\n❌ Failed to create comparison. Check the error messages above.")

if __name__ == "__main__":
    main() 