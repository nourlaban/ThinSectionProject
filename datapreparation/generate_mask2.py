import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS
from typing import Dict, Tuple, Any, NamedTuple
from dataclasses import dataclass

@dataclass
class RasterParams:
    width: int
    height: int
    nodata_mask: np.ndarray
    transform: Any
    crs: Any

@dataclass
class CategoryMaps:
    category_map: Dict[Any, int]
    class_name_map: Dict[Any, str]

def load_vector_data(shapefile_path: str) -> gpd.GeoDataFrame:
    return gpd.read_file(shapefile_path)

def get_raster_parameters(dataset_path: str) -> RasterParams:
    with rasterio.open(dataset_path, 'r') as dst:
        origin_x, origin_y = dst.transform * (0, 0)
        pixel_size_x, pixel_size_y = dst.transform[0], -dst.transform[4]
        imdata = dst.read(6)
        return RasterParams(
            width=int(dst.width),
            height=int(dst.height),
            nodata_mask=imdata == 0,
            transform=dst.transform,
            crs=dst.crs
        )

def create_category_maps(
    vector_data: gpd.GeoDataFrame,
    class_value_attribute: str = 'gridcode',
    class_name_attribute: str = 'Class'
) -> CategoryMaps:
    if class_value_attribute not in vector_data.columns:
        raise ValueError(f"Attribute '{class_value_attribute}' not found in shapefile.")
    if class_name_attribute not in vector_data.columns:
        raise ValueError(f"Attribute '{class_name_attribute}' not found in shapefile.")
    
    categories = vector_data[class_value_attribute].unique()
    category_map = {category: idx + 1 for idx, category in enumerate(categories)}
    class_name_map = {
        category: vector_data[vector_data[class_value_attribute] == category][class_name_attribute].iloc[0]
        for category in categories
    }
    
    return CategoryMaps(category_map, class_name_map)

def rasterize_vector_data(
    vector_data: gpd.GeoDataFrame,
    raster_params: RasterParams,
    category_maps: CategoryMaps,
    output_path: str
) -> None:
    metadata = {
        'driver': 'GTiff',
        'count': 1,
        'dtype': 'uint8',
        'width': raster_params.width,
        'height': raster_params.height,
        'crs': raster_params.crs,
        'transform': raster_params.transform
    }

    try:
        with rasterio.open(output_path, 'w', **metadata) as dst:
            out_image = rasterize(
                [(geometry,11 if value == 100 else value)
                 for value, geometry in zip(vector_data['gridcode'], vector_data.geometry)],
                out_shape=(raster_params.height, raster_params.width),
                transform=raster_params.transform,
                fill=0,
                dtype='uint8'
            )           
            dst.write(out_image, 1)

        print(f"Shapefile rasterized with categorical data and saved as {output_path}")

    except rasterio.errors.RasterioIOError as e:
        print(f"RasterioIOError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

def generate_mask(shapefile: str, tif_file: str, mask_file: str) -> None:
    vector_data = load_vector_data(shapefile)
    raster_params = get_raster_parameters(tif_file)
    category_maps = create_category_maps(vector_data)
    rasterize_vector_data(vector_data, raster_params, category_maps, mask_file)