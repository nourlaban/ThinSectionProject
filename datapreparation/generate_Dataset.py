import h5py
import rasterio
import numpy as np
import pyproj
from rasterio.transform import from_origin

class HyperspectralImageReader:
    def __init__(self, filename):
        self.filename = filename
        self.h5file = None
        self.datasetVNIR = None
        self.datasetSWIR = None
        self.metadata = None
    
    def latlon_to_epsg32636(self, latitude, longitude,utmcode=32636 ):
        """Converts latitude and longitude coordinates to EPSG:32636.

        Args:
            latitude: The latitude coordinate.
            longitude: The longitude coordinate.

        Returns:
            A tuple containing the easting and northing in EPSG:32636.
            Returns None if the input coordinates are invalid.
        """
        try:
            # Define the source CRS (WGS 84 - lat/lon)
            wgs84 = pyproj.CRS("EPSG:4326")

            # Define the target CRS (EPSG:32636)
            utm36n = pyproj.CRS("EPSG:"+str(utmcode))

            # Create a transformer object
            transformer = pyproj.Transformer.from_crs(wgs84, utm36n, always_xy=True)

            # Perform the transformation
            easting, northing = transformer.transform(longitude, latitude)

            return easting, northing

        except Exception as e:
            print(f"Error during coordinate transformation: {e}")
            return None

    
    def open_file(self):
        self.h5file = h5py.File(self.filename, 'r')
        datasetVNIR_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/VNIR_Cube'
        datasetSWIR_path = '/HDFEOS/SWATHS/PRS_L2D_HCO/Data Fields/SWIR_Cube'
       
        self.datasetVNIR = self.h5file[datasetVNIR_path]
        self.datasetSWIR = self.h5file[datasetSWIR_path]

    def get_metadata(self):
        dataset_lat = '/HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Latitude'
        dataset_lon = '/HDFEOS/SWATHS/PRS_L2D_HCO/Geolocation Fields/Longitude'
        self.dataset_lat = self.h5file[dataset_lat]
        data = self.dataset_lat[:]
        minlat = data[0,0] #np.min(data), 
        maxlat = data[data.shape[0]-1,data.shape[1]-1] #np.max(data)

        self.dataset_lon = self.h5file[dataset_lon]
        data = self.dataset_lon[:]
        minlon = data[0,0] 
        maxlon = data[data.shape[0]-1,data.shape[1]-1]
        mineasting, minnorthing =  self.latlon_to_epsg32636(minlat,minlon)
        maxeasting, maxnorthing =  self.latlon_to_epsg32636(maxlat,maxlon)

        origin_x= float( mineasting )  #659247.5
        max_x   = float(maxeasting) #698097.4382495880126953
        max_y   = float(maxnorthing) # 2706767.5
        origin_y= float(minnorthing) #2743007.5

        pixel_size_x=(max_x - origin_x) / self.datasetVNIR.shape[2]
        pixel_size_y=(origin_y - max_y) / self.datasetVNIR.shape[0]

        return origin_x, origin_y, pixel_size_x, pixel_size_y

    def set_metadata(self):
        origin_x, origin_y, pixel_size_x, pixel_size_y = self.get_metadata()
        transform = from_origin(origin_x, origin_y, pixel_size_x, pixel_size_y)
        vnir_bands = self.datasetVNIR.shape[1] - 5
        swir_bands = self.datasetSWIR.shape[1] - len(self.excluded_swir_bands())
        total_bands = vnir_bands + swir_bands
        self.metadata = {
            'driver': 'GTiff',
            'height': self.datasetVNIR.shape[0],
            'width': self.datasetVNIR.shape[2],
            'count': total_bands,
            'dtype': str(self.datasetVNIR.dtype),
            'crs': {'init': 'epsg:32636'},
            'transform': transform
        }

    def excluded_swir_bands(self):
        return list(range(3, 6)) + list(range(40, 57)) + list(range(86, 112)) + list(range(152, 172))

    def write_to_geotiff(self, output_path):
        with rasterio.open(output_path, 'w', **self.metadata) as dst:
            # Write VNIR bands
            for i in range(self.datasetVNIR.shape[1] - 5):
                dst.write(self.datasetVNIR[:, i + 5, :], i + 1)
            
            # Write SWIR bands, excluding specified ones
            current_band = self.datasetVNIR.shape[1] - 5
            for j in range(self.datasetSWIR.shape[1]):
                if j not in self.excluded_swir_bands():
                    dst.write(self.datasetSWIR[:, j, :], current_band + 1)
                    current_band += 1

    def close(self):
        self.h5file.close()

# Example usage
# data_dir = r'D:\work\PRISMA Group\prisma_hyperspecrtal_data_classification\data\rawdata\\'
# h5_file =  data_dir + 'PRS_L2D_STD_20200725083506_20200725083510_0001.he5'
# tif_file = data_dir +   'combined_hyperspectral_image_VNIR_SWIR2.tif'


def generate_tif(h5_file,tif_file):

    reader = HyperspectralImageReader(h5_file)
    reader.open_file()
    reader.set_metadata()
    reader.write_to_geotiff(tif_file)
    reader.close()




if __name__ == "__main__":
    data_dir = r'D:\work\PRISMA Group\00 project\data\Seenaa02\PRS_L2D_STD_20201020083329_20201020083333_0001\\'
    h5_file =  data_dir + 'PRS_L2D_STD_20201020083329_20201020083333_0001.he5'
    tif_file = data_dir +   'PRS_L2D_STD_20201020083329_20201020083333_0001.tif'


    reader = HyperspectralImageReader(h5_file)
    reader.open_file()
    reader.set_metadata()
    reader.write_to_geotiff(tif_file)
    reader.close()
