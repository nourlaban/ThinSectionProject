import geopandas as gpd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape

def create_shapefile_from_mask(mask_file, output_shapefile):
    # Example usage:
    class_name_map = {
        1: 'alkali feldspar granite',
        2: 'gabbro',
        3: 'gneiss',
        4: 'granodiorite',
        5: 'massive metavolcanics',
        6: 'Monzogranite',
        7: 'pigmatitic biotite granite',
        8: 'Riebeckite granite-syenite--alkali volcanics',
        9: 'sedimentary rock',
        10: 'wadi deposits',
        11: 'unclassified',
        12: 'backgro'
    }

    with rasterio.open(mask_file) as src:
        image = src.read(1)  # Read the first band
        mask = image != 0  # Exclude zero values if they are nodata

        results = (
            {'properties': {'class_value': int(v), 'class_name': class_name_map[v]},
             'geometry': shape(s)}
            for i, (s, v) in enumerate(shapes(image, mask=mask, transform=src.transform))
            if v in class_name_map
        )

        geoms = list(results)

        # Create a GeoDataFrame
        gdf = gpd.GeoDataFrame.from_features(geoms, crs=src.crs)

        # Ensure the geometry column is correctly set
        gdf.set_geometry('geometry', inplace=True)

        # Save to a new shapefile
        gdf.to_file(output_shapefile)


# create_shapefile_from_mask('path/to/output_mask.tif', 'path/to/new_shapefile.shp', class_name_map)