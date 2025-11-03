import os
import cv2
import staintools

"""
Simple batch stain normalization script.
- Loads a reference image
- Applies Vahadane stain normalization
- Preserves folder structure while writing results
"""

# -----------------------------------------------------------------------------
# User paths — adjust as needed
# -----------------------------------------------------------------------------

ref_img_path = "C:/Users/jaswa/Downloads/TCGA-95-8494-01Z-00-DX1.716299EF-71BB-4095-8F4D-F0C2252CE594_5932_5708_0.png"
input_root_dir = "C:/Users/jaswa/Desktop/Stage IV_Patches"
output_root_dir = "C:/Users/jaswa/Desktop/Stage IV_normalized_Patches"

# -----------------------------------------------------------------------------
# Load reference and set up normalizer
# -----------------------------------------------------------------------------

print("Loading reference image...")
ref_img = staintools.read_image(ref_img_path)
ref_img = staintools.LuminosityStandardizer.standardize(ref_img)

normalizer = staintools.StainNormalizer(method="vahadane")
normalizer.fit(ref_img)
print("Reference fit complete.")

# -----------------------------------------------------------------------------
# Normalize all patches and mirror directory structure
# -----------------------------------------------------------------------------

for subdir, _, files in os.walk(input_root_dir):
    for file in files:
        if file.lower().endswith(".png"):
            in_path = os.path.join(subdir, file)

            try:
                # Read + standardize patch
                img = staintools.read_image(in_path)
                img = staintools.LuminosityStandardizer.standardize(img)

                # Normalize stain
                norm_img = normalizer.transform(img)

                # Determine output path preserving folder tree
                rel = os.path.relpath(in_path, input_root_dir)
                out_path = os.path.join(output_root_dir, rel)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)

                # Save result
                cv2.imwrite(out_path, cv2.cvtColor(norm_img, cv2.COLOR_RGB2BGR))
                print(f"Normalized: {rel}")

            except Exception as exc:
                print(f"Failed on: {in_path}")
                print(f"Reason: {exc}")

print("All patches processed.")
