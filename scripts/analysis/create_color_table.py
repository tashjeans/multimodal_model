#!/usr/bin/env python3
"""
Create a professional color label table for Native vs Boltz structure comparison
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os

def create_color_table():
    """
    Create a professional color label table
    """
    try:
        # Create output directory
        output_dir = "color_tables"
        os.makedirs(output_dir, exist_ok=True)
        
        print("🎨 Creating professional color label table...")
        
        # Define the color scheme from your working script
        chains = ["A", "B", "C", "D", "E"]
        native_colors = ["forest", "cyan", "blue", "green", "purple"]
        test_colors = ["lime", "yellow", "grey", "orange", "pink"]
        
        # Create a pandas DataFrame for easy manipulation
        df = pd.DataFrame({
            'Chain': chains,
            'Native Color': native_colors,
            'Boltz Color': test_colors,
            'Native Description': ['Main TCR α chain', 'Main TCR β chain', 'Short peptide', 'HLA-A*11:01 α chain', 'HLA-A*11:01 β2m'],
            'Boltz Description': ['Predicted TCR α chain', 'Predicted TCR β chain', 'Predicted peptide', 'Predicted HLA α chain', 'Predicted HLA β2m']
        })
        
        # Save as CSV
        csv_path = os.path.join(output_dir, "color_label_table.csv")
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV table saved as: {csv_path}")
        
        # Create matplotlib table
        create_matplotlib_table(df, output_dir)
        
        # Create PIL image table
        create_pil_table(df, output_dir)
        
        # Create publication-ready table
        create_publication_table(df, output_dir)
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating color table: {e}")
        return False

def create_matplotlib_table(df, output_dir):
    """
    Create a matplotlib-based color table
    """
    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table
        table = ax.table(cellText=df.values, 
                        colLabels=df.columns,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)
        
        # Color the header
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color the chain column
        for i in range(1, len(df) + 1):
            table[(i, 0)].set_facecolor('#E8F5E8')
            table[(i, 0)].set_text_props(weight='bold')
        
        # Add title
        plt.title('Molecular Structure Color Coding: Native PDB vs Boltz Prediction', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save
        table_path = os.path.join(output_dir, "color_table_matplotlib.png")
        plt.savefig(table_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matplotlib table saved as: {table_path}")
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️  Matplotlib table creation failed: {e}")

def create_pil_table(df, output_dir):
    """
    Create a PIL-based color table with actual color swatches
    """
    try:
        # Define color mappings (PIL color names)
        color_mapping = {
            'forest': '#228B22',
            'cyan': '#00FFFF', 
            'blue': '#0000FF',
            'green': '#008000',
            'purple': '#800080',
            'lime': '#00FF00',
            'yellow': '#FFFF00',
            'grey': '#808080',
            'orange': '#FFA500',
            'pink': '#FFC0CB'
        }
        
        # Create image
        img_width, img_height = 1200, 800
        img = Image.new('RGB', (img_width, img_height), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to load fonts
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        except:
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw title
        title = "Molecular Structure Color Coding: Native PDB vs Boltz Prediction"
        draw.text((img_width//2, 30), title, fill='black', font=font_large, anchor='mm')
        
        # Draw table headers
        headers = ['Chain', 'Native Color', 'Boltz Color', 'Native Description', 'Boltz Description']
        col_widths = [100, 150, 150, 300, 300]
        start_x = 50
        
        # Draw header row
        y_pos = 100
        for i, header in enumerate(headers):
            x_pos = start_x + sum(col_widths[:i])
            # Header background
            draw.rectangle([x_pos, y_pos, x_pos + col_widths[i], y_pos + 40], 
                          fill='#4CAF50', outline='black', width=2)
            # Header text
            draw.text((x_pos + col_widths[i]//2, y_pos + 20), header, 
                     fill='white', font=font_medium, anchor='mm')
        
        # Draw data rows
        y_pos = 140
        for idx, row in df.iterrows():
            for i, (col, value) in enumerate(row.items()):
                x_pos = start_x + sum(col_widths[:i])
                
                # Cell background
                if i == 0:  # Chain column
                    bg_color = '#E8F5E8'
                else:
                    bg_color = 'white'
                
                draw.rectangle([x_pos, y_pos, x_pos + col_widths[i], y_pos + 50], 
                              fill=bg_color, outline='black', width=1)
                
                # Draw color swatches for color columns
                if i == 1:  # Native color
                    color_hex = color_mapping.get(value, '#000000')
                    draw.rectangle([x_pos + 10, y_pos + 10, x_pos + 40, y_pos + 40], 
                                  fill=color_hex, outline='black', width=2)
                    draw.text((x_pos + 60, y_pos + 25), value, fill='black', font=font_small, anchor='lm')
                elif i == 2:  # Boltz color
                    color_hex = color_mapping.get(value, '#000000')
                    draw.rectangle([x_pos + 10, y_pos + 10, x_pos + 40, y_pos + 40], 
                                  fill=color_hex, outline='black', width=2)
                    draw.text((x_pos + 60, y_pos + 25), value, fill='black', font=font_small, anchor='lm')
                else:
                    # Regular text
                    draw.text((x_pos + 10, y_pos + 25), str(value), 
                             fill='black', font=font_small, anchor='lm')
            
            y_pos += 50
        
        # Add legend
        legend_y = y_pos + 20
        draw.text((img_width//2, legend_y), "Color Legend", 
                 fill='black', font=font_medium, anchor='mm')
        
        # Save
        table_path = os.path.join(output_dir, "color_table_pil.png")
        img.save(table_path, 'PNG', dpi=(300, 300))
        print(f"✅ PIL table saved as: {table_path}")
        
    except Exception as e:
        print(f"⚠️  PIL table creation failed: {e}")

def create_publication_table(df, output_dir):
    """
    Create a publication-ready LaTeX table
    """
    try:
        # Create LaTeX table
        latex_table = r"""
\begin{table}[h]
\centering
\caption{Molecular Structure Color Coding: Native PDB vs Boltz Prediction}
\label{tab:structure_colors}
\begin{tabular}{|c|c|c|p{3cm}|p{3cm}|}
\hline
\textbf{Chain} & \textbf{Native Color} & \textbf{Boltz Color} & \textbf{Native Description} & \textbf{Boltz Description} \\
\hline
"""
        
        # Add data rows
        for _, row in df.iterrows():
            latex_table += f"{row['Chain']} & {row['Native Color']} & {row['Boltz Color']} & {row['Native Description']} & {row['Boltz Description']} \\\\\n"
        
        latex_table += r"""
\hline
\end{tabular}
\end{table}
"""
        
        # Save LaTeX table
        latex_path = os.path.join(output_dir, "color_table_latex.tex")
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        
        print(f"✅ LaTeX table saved as: {latex_path}")
        
        # Also create a simple text version
        text_table = "Molecular Structure Color Coding: Native PDB vs Boltz Prediction\n"
        text_table += "=" * 80 + "\n\n"
        text_table += f"{'Chain':<6} {'Native':<12} {'Boltz':<12} {'Native Description':<25} {'Boltz Description':<25}\n"
        text_table += "-" * 80 + "\n"
        
        for _, row in df.iterrows():
            text_table += f"{row['Chain']:<6} {row['Native Color']:<12} {row['Boltz Color']:<12} {row['Native Description']:<25} {row['Boltz Description']:<25}\n"
        
        text_path = os.path.join(output_dir, "color_table_text.txt")
        with open(text_path, 'w') as f:
            f.write(text_table)
        
        print(f"✅ Text table saved as: {text_path}")
        
    except Exception as e:
        print(f"⚠️  Publication table creation failed: {e}")

def main():
    """Main function"""
    print("🎨 Creating Professional Color Label Tables")
    print("=" * 60)
    print("This will create multiple formats of your color coding table:")
    print("✅ CSV format (for data analysis)")
    print("✅ PNG images (for presentations)")
    print("✅ LaTeX format (for publications)")
    print("✅ Text format (for documentation)")
    print("=" * 60)
    
    # Create the color tables
    success = create_color_table()
    
    if success:
        print("\n🎉 Color label tables created successfully!")
        print("📁 Check the 'color_tables' folder for your tables:")
        print("  - color_label_table.csv (data format)")
        print("  - color_table_matplotlib.png (image format)")
        print("  - color_table_pil.png (image with color swatches)")
        print("  - color_table_latex.tex (publication format)")
        print("  - color_table_text.txt (text format)")
        print("\n📊 These tables clearly show the color coding for:")
        print("  - Chain A: Main TCR chains")
        print("  - Chain B: Secondary TCR chains") 
        print("  - Chain C: Peptide chains")
        print("  - Chain D: HLA α chains")
        print("  - Chain E: HLA β2m chains")
    else:
        print("\n❌ Failed to create color tables. Check the error messages above.")

if __name__ == "__main__":
    main() 