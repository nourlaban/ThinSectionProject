from generate_mask import * 
from tilling_recombine import *
from learnunet_pytorch import *


def generate_masking(tif_file,shapefile,mask_file):   

    generate_mask(shapefile, tif_file, mask_file)
           
def tilling(data_dir,image_path,mask_path,output_directory,recombine_image,recombine_mask , task = 'create' ):   
    
    tile_size = 32

    tiler = ImageTiler(image_path, mask_path, tile_size, output_directory)
    if task == 'create': 
        tiler.create_tiles()

        
    if task == 'combine':
        tiler.recombine_tiles(recombine_image ,recombine_mask)
     
def learnunet(output_directory,result_dir,test_only = False):

        # Example usage
    datadir = output_directory
    num_channels = 6  # Adjust based on your hyperspectral data
    num_classes = 12  # Adjust based on your number of classes
    input_shape = (32, 32)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #unet_model = UNet(n_channels=num_channels, n_classes=num_classes)
    
   
    # Usage
    # unet_model = ResNetUNet_withBatch_dropout(n_channels=num_channels, n_classes=num_classes)  # Now works with any number of input channels
    unet_model = ResNext101UNet(n_channels=num_channels, n_classes=num_classes)  # Now works with any number of input channels
    #unet_model = ResNetUNet(n_channels=num_channels, n_classes=num_classes)  # Now works with any number of input channels



    # Define data augmentation pipeline
    augmentation_pipeline = Compose([
        RandomHorizontalFlip(),
        RandomVerticalFlip(),
        RandomRotation(degrees=5)
    ])

    # Usage with data augmentation
    train_dataset = MulticlassHyperspectralDataset(
        os.path.join(datadir, 'train', 'images'),
        os.path.join(datadir, 'train', 'masks'),
        num_classes=num_classes,
        transform=Resize(input_shape),
        augmentation=augmentation_pipeline
    )

    val_dataset = MulticlassHyperspectralDataset(
        os.path.join(datadir, 'val', 'images'),
        os.path.join(datadir, 'val', 'masks'),
        num_classes=num_classes,
        transform=Resize(input_shape)
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    if not test_only: 
        ##train_model_lr(unet_model, train_loader, val_loader, device, num_classes, epochs=200)
        train_model_lr(unet_model, train_loader, val_loader,result_dir, device, num_classes, epochs=30, learning_rate=5e-4)
    else:
    # Load the state dictionary
        unet_model.load_state_dict(torch.load(r'output/a013/results/trained_model_imestamp.pth'))

        
    # Predict and save
    test_image_dir = os.path.join(datadir, 'all_tiles', 'images')
    output_dir = os.path.join(datadir, 'all_tiles', 'predictions')
    predict_and_save(unet_model, test_image_dir, output_dir, device, input_shape, num_classes)

def trainonly():
    pass
def testonly():
    pass


if __name__ == "__main__":
    data_dir    = r'D:\narssprojects\thensections\6bands\data' 

    image_dir   =  r'D:\narssprojects\thensections\6bands\data\images'
    maskshp_dir =  r'D:\narssprojects\thensections\6bands\data\annotationshapes'

    temp_intermediate_dir = r'D:\narssprojects\thensections\6bands\intermediateResults'
    os.makedirs(temp_intermediate_dir, exist_ok=True)





    tif_file   = r'D:\narssprojects\thensections\6bands\data\store\images\Boitite_Gr_clip.tif'
    shapefile  = r'D:\narssprojects\thensections\6bands\data\store\annotationshapes\Boitite_Gr_clip.shp' 
    
    output_dir = r'D:\narssprojects\thensections\output'
    
    experimemnt = 'a014'


    experiment_output_dir = os.path.join(output_dir,experimemnt)
    os.makedirs(experiment_output_dir, exist_ok=True)
    test_only = True

    mask_file  = os.path.join( experiment_output_dir,'Boitite_Gr_clip_shpmask.tif')  
    output_directory = os.path.join(experiment_output_dir, 'output_tiles')
    recombine_image =  os.path.join(experiment_output_dir, 'Boitite_Gr_clip_recombined_image.tif')
    recombine_mask  =   os.path.join(experiment_output_dir, 'Boitite_Gr_clip_recombined_mask.tif')
    result_dir      =  os.path.join(experiment_output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    
    test_only = False

    if not test_only:
        generate_masking(tif_file,shapefile,mask_file)
        tilling(data_dir,tif_file,mask_file,output_directory,recombine_image,recombine_mask, task = 'create')
        learnunet(output_directory,result_dir,test_only = False)
        tilling(data_dir,tif_file,mask_file,output_directory,recombine_image,recombine_mask, task = 'combine')
    else:
        learnunet(output_directory,result_dir,test_only = True)
        tilling(data_dir,tif_file,mask_file,output_directory,recombine_image,recombine_mask, task = 'combine')




    


    
