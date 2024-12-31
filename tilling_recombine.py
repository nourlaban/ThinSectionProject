import os
import rasterio
import numpy as np
from rasterio.windows import Window
from rasterio.transform import from_origin
import random
from glob import glob

class ImageTiler:
    def __init__(self, image_path, mask_path, tile_size, output_dir,prefix='tile', train_ratio=0.7, val_ratio=0.15):
        self.image_path = image_path
        self.mask_path = mask_path
        self.tile_size = tile_size
        self.output_dir = output_dir
        self.prefix = prefix
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.map_file = os.path.join(output_dir, 'map.txt')
        self.transform_file = os.path.join(output_dir, 'transforms.txt')

    def create_tiles(self):
        os.makedirs(self.output_dir, exist_ok=True)

        for subset in ['train', 'val', 'test', 'all_tiles']:
            images_dir = os.path.join(self.output_dir, subset, 'images')
            masks_dir = os.path.join(self.output_dir, subset, 'masks')
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

        with rasterio.open(self.image_path) as src_img:
            img_width = src_img.width
            img_height = src_img.height
            num_tiles_x = (img_width + self.tile_size - 1) // self.tile_size
            num_tiles_y = (img_height + self.tile_size - 1) // self.tile_size
            total_tiles = num_tiles_x * num_tiles_y

            indices = [(i, j) for i in range(0, img_height, self.tile_size) for j in range(0, img_width, self.tile_size)]
            random.shuffle(indices)

            num_train = int(total_tiles * self.train_ratio)
            num_val = int(total_tiles * self.val_ratio)

            train_indices = indices[:num_train]
            val_indices = indices[num_train:num_train + num_val]
            test_indices = indices[num_train + num_val:]

        with open(self.map_file, 'w') as map_file, open(self.transform_file, 'w') as transform_file:
            with rasterio.open(self.image_path) as src_img, rasterio.open(self.mask_path) as src_mask:
                tile_count = 0

                for idx, (i, j) in enumerate(indices):
                    all_tiles = "all_tiles"
                    if (i, j) in train_indices:
                        subset = 'train'
                    elif (i, j) in val_indices:
                        subset = 'val'
                    else:
                        subset = 'test'

                    images_dir = os.path.join(self.output_dir, subset, 'images')
                    all_images_dir = os.path.join(self.output_dir, all_tiles, 'images')
                    masks_dir = os.path.join(self.output_dir, subset, 'masks')
                    all_masks_dir = os.path.join(self.output_dir, all_tiles, 'masks')

                    window = Window(j, i, self.tile_size, self.tile_size)
                    img_tile = src_img.read(window=window)
                    mask_tile = src_mask.read(window=window)

                    pad_height = max(0, self.tile_size - img_tile.shape[1])
                    pad_width = max(0, self.tile_size - img_tile.shape[2])

                    if pad_height > 0 or pad_width > 0:
                        img_tile = np.pad(img_tile, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
                        mask_tile = np.pad(mask_tile, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

                    transform = from_origin(
                        src_img.transform.c + j * src_img.transform.a,
                        src_img.transform.f + i * src_img.transform.e,
                        src_img.transform.a,
                        src_img.transform.e
                    )
                    img_meta = src_img.meta.copy()
                    img_meta.update({
                        'height': self.tile_size,
                        'width': self.tile_size,
                        'transform': transform
                    })

                    mask_meta = src_mask.meta.copy()
                    mask_meta.update({
                        'height': self.tile_size,
                        'width': self.tile_size,
                        'transform': transform
                    })

                    img_tile_path = os.path.join(images_dir, f"{self.prefix}_img_{i}_{j}.tif")
                    img_tile_path2 = os.path.join(all_images_dir, f"{self.prefix}_img_{i}_{j}.tif")
                    with rasterio.open(img_tile_path, 'w', **img_meta) as dest:
                        dest.write(img_tile)
                    with rasterio.open(img_tile_path2, 'w', **img_meta) as dest:
                        dest.write(img_tile)

                    mask_tile_path = os.path.join(masks_dir, f"{self.prefix}_mask_{i}_{j}.tif")
                    mask_tile_path2 = os.path.join(all_masks_dir, f"{self.prefix}_mask_{i}_{j}.tif")
                    with rasterio.open(mask_tile_path, 'w', **mask_meta) as dest:
                        dest.write(mask_tile)
                    with rasterio.open(mask_tile_path2, 'w', **mask_meta) as dest:
                        dest.write(mask_tile)

                    map_file.write(f"{img_tile_path} {mask_tile_path}\n")
                    transform_file.write(f"{img_tile_path} {transform}\n")

                    tile_count += 1

        print(f"Created {tile_count} tiles and map.txt.")

    def recombine_tiles(self, output_image_path, output_mask_path):
        all_images_dir = os.path.join(self.output_dir, 'all_tiles', 'images')
        all_masks_dir = os.path.join(self.output_dir, 'all_tiles', 'predictions')
        image_tile_files = glob(os.path.join(all_images_dir, f"{self.prefix}_img_*.tif"))
        mask_tile_files = glob(os.path.join(all_masks_dir, f"*.tif"))

        with rasterio.open(self.image_path) as src_img:
            full_image = np.zeros((src_img.count, src_img.height, src_img.width), dtype=src_img.dtypes[0])

        with rasterio.open(self.mask_path) as src_mask:
            full_mask = np.zeros((src_mask.count, src_mask.height, src_mask.width), dtype=src_mask.dtypes[0])

        for tile_file in image_tile_files:
            _, filename = os.path.split(tile_file)
            _, _, start_y, start_x = filename.rstrip('.tif').split('_')

            start_x, start_y = int(start_x), int(start_y)

            with rasterio.open(tile_file) as tile:
                tile_data = tile.read()
                end_y = min(start_y + tile_data.shape[1], full_image.shape[1])
                end_x = min(start_x + tile_data.shape[2], full_image.shape[2])

                full_image[:, start_y:end_y, start_x:end_x] = tile_data[:, :end_y-start_y, :end_x-start_x]

        for tile_file in mask_tile_files:
            _, filename = os.path.split(tile_file)
            _, _, _, start_y, start_x = filename.rstrip('.tif').split('_')

            start_x, start_y = int(start_x), int(start_y)

            with rasterio.open(tile_file) as tile:
                tile_data = tile.read()
                end_y = min(start_y + tile_data.shape[1], full_mask.shape[1])
                end_x = min(start_x + tile_data.shape[2], full_mask.shape[2])

                full_mask[:, start_y:end_y, start_x:end_x] = tile_data[:, :end_y-start_y, :end_x-start_x]

        image_meta = src_img.meta.copy()
        image_meta.update({
            'height': src_img.height,
            'width': src_img.width,
            'transform': src_img.transform
        })

        mask_meta = src_mask.meta.copy()
        mask_meta.update({
            'height': src_mask.height,
            'width': src_mask.width,
            'transform': src_mask.transform
        })

        with rasterio.open(output_image_path, 'w', **image_meta) as dest:
            dest.write(full_image)

        with rasterio.open(output_mask_path, 'w', **mask_meta) as dest:
            dest.write(full_mask)

        print(f"Recombined image saved at {output_image_path}")
        print(f"Recombined mask saved at {output_mask_path}")

# Example usage

if __name__ == "__main__":
    data_dir = r'F:\Senaa\thensections\6bands'
    image_path = os.path.join(data_dir, 'Boitite_Gr_clip.tif')
    mask_path = os.path.join(data_dir, 'Boitite_Gr_clip_new.tif')
    output_directory = os.path.join(data_dir, 'output_tiles')
    tile_size = 64

    tiler = ImageTiler(image_path, mask_path, tile_size, output_directory)
    tiler.create_tiles()
    # tiler.recombine_tiles(
    #     os.path.join(data_dir, 'recombined_image444.tif'),
    #     os.path.join(data_dir, 'recombined_mask555.tif')
    # )
