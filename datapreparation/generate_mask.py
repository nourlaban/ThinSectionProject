import geopandas as gpd
import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS

class Rasterizer:
    def __init__(self, shapefile_path, dataset_path, output_raster_path, pixel_size=30):
        self.shapefile_path = shapefile_path
        self.dataset_path = dataset_path
        self.output_raster_path = output_raster_path
        self.pixel_size = pixel_size
        self.vector_data = None
        self.crs = None
        self.transform = None
        self.width = None
        self.height = None
        self.nodata_mask = None
        self.category_map = None

    def load_vector_data(self):
        self.vector_data = gpd.read_file(self.shapefile_path)
        if self.vector_data.crs is None:
            raise ValueError("The shapefile does not have a CRS defined.")
        self.crs = self.vector_data.crs.to_string()

    def setup_raster_parameters(self):
        with rasterio.open(self.dataset_path, 'r') as dst:
            origin_x, origin_y = dst.transform * (0, 0)
            pixel_size_x, pixel_size_y = dst.transform[0], -dst.transform[4]
            imdata = dst.read(6)
            self.width = int(dst.width)
            self.height = int(dst.height)
            self.nodata_mask = imdata == 0
            self.transform = from_origin(origin_x, origin_y, pixel_size_x, pixel_size_y)

    def create_category_map(self, attribute='Classvalue'):
        if attribute not in self.vector_data.columns:
            raise ValueError(f"Attribute '{attribute}' not found in shapefile.")
        categories = self.vector_data[attribute].unique()
        self.category_map = {category: idx + 1 for idx, category in enumerate(categories)}

    def rasterize_vector_data(self):
        metadata = {
            'driver': 'GTiff',
            'count': 1,
            'dtype': 'uint8',
            'width': self.width,
            'height': self.height,
            'crs': self.crs,
            'transform': self.transform
        }

        try:
            with rasterio.open(self.output_raster_path, 'w', **metadata) as dst:
                out_image = rasterize(
                    [(geometry, self.category_map[value])
                     for value, geometry in zip(self.vector_data['Classvalue'], self.vector_data.geometry)],
                    out_shape=(self.height, self.width),
                    transform=self.transform,
                    fill=11,
                    dtype='uint8'
                )
                out_image[self.nodata_mask] = 0
                dst.write(out_image, 1)

            print(f"Shapefile rasterized with categorical data and saved as {self.output_raster_path}")

        except rasterio.errors.RasterioIOError as e:
            print(f"RasterioIOError: {e}")

        except Exception as e:
            print(f"An error occurred: {e}")



def  generate_mask(shapefile, tif_file, mask_file): 
    rasterizer = Rasterizer(shapefile, tif_file, mask_file)
    rasterizer.load_vector_data()
    rasterizer.setup_raster_parameters()
    rasterizer.create_category_map()
    rasterizer.rasterize_vector_data()
