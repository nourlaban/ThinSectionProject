import rasterio
from rasterio.mask import mask
import numpy as np
import geopandas as gpd
from shapely.geometry import mapping

workdir = r"D:\work\ThinSections\newproject\6bands\\"
# Load the TIFF image
with rasterio.open(workdir + "Boitite_Gr_clip_mask.tif" ) as src:
    # Read the image data
    image = src.read(1)
    
    # Example: Change all values above a certain threshold
    threshold = 100
    image[image == threshold] = 11
    
    # Save the modified image
    profile = src.profile
    with rasterio.open(workdir + "Boitite_Gr_clip_mask_recoded.tif", 'w', **profile) as dst:
        dst.write(image, 1)

# # Optionally, apply a mask using a shapefile
# gdf = gpd.read_file('path/to/your/shapefile.shp')
# geoms = [mapping(geom) for geom in gdf.geometry]

# with rasterio.open('path/to/your/image.tif') as src:
#     out_image, out_transform = mask(src, geoms, crop=True)
#     out_meta = src.meta.copy()

# out_meta.update({"driver": "GTiff",
#                  "height": out_image.shape[1],
#                  "width": out_image.shape[2],
#                  "transform": out_transform})

# with rasterio.open('path/to/your/masked_image.tif', 'w', **out_meta) as dest:
#     dest.write(out_image)
