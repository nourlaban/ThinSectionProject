from generate_mask import * 
from tilling_recombine import *
from learnunet_pytorch import *


def generate_masking(tif_file,shapefile,mask_file):   

    generate_mask(shapefile, tif_file, mask_file)
   
def learnunet(output_directory,result_dir):
        # Example usage
    datadir = output_directory
    num_channels = 6  # Adjust based on your hyperspectral data
    num_classes = 12  # Adjust based on your number of classes
    input_shape = (32, 32)   
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
    ##train_model_lr(unet_model, train_loader, val_loader, device, num_classes, epochs=200)
    train_model_lr(unet_model, train_loader, val_loader,result_dir, device, num_classes, epochs=30, learning_rate=5e-4)
   

        
    


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir    = r'D:\narssprojects\thensections\6bands\data\group1'     
    tile_size   =  32
    num_channels = 6  # Adjust based on your hyperspectral data
    num_classes = 12  # Adjust based on your number of classes

    temp_intermediate_dir = r'D:\narssprojects\thensections\6bands\intermediateResults'
    os.makedirs(temp_intermediate_dir, exist_ok=True)

    output_dir = r'D:\narssprojects\thensections\output'
    experimemnt = 'a01'
    experiment_output_dir = os.path.join(output_dir,experimemnt)
    os.makedirs(experiment_output_dir, exist_ok=True)
    result_dir      =  os.path.join(experiment_output_dir, 'results')
    os.makedirs(result_dir, exist_ok=True)
    output_directory = os.path.join(experiment_output_dir, 'output_tiles')
    os.makedirs(output_directory, exist_ok=True)

    output_test_directory = os.path.join(experiment_output_dir, 'output_test_tiles')
    os.makedirs(output_test_directory, exist_ok=True)



    def generate_dataset():
        image_dir   =  os.path.join(data_dir,'images') 
        maskshp_dir =   os.path.join(data_dir,'shapes') 
        for filename in os.listdir(image_dir):    
            if filename.endswith(".tif"):
                tif_name= os.path.splitext(filename)[0]
                tif_file   = os.path.join(image_dir,tif_name+'.tif' ) 
                shapefile  = os.path.join(maskshp_dir,"Vec_"+tif_name+'.shp' )
                mask_file  = os.path.join( experiment_output_dir,tif_name + '_shpmask.tif')                  
                generate_masking(tif_file,shapefile,mask_file)                   
                create_tiles(tif_file, mask_file , tile_size, output_directory, prefix=imagename)           
            
    def train_deep_learning():              
                  
       
        input_shape = (tile_size, tile_size)   
       
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
            os.path.join(output_directory, 'train', 'images'),
            os.path.join(output_directory, 'train', 'masks'),
            num_classes=num_classes,
            transform=Resize(input_shape),
            augmentation=augmentation_pipeline
        )

        val_dataset = MulticlassHyperspectralDataset(
            os.path.join(output_directory, 'val', 'images'),
            os.path.join(output_directory, 'val', 'masks'),
            num_classes=num_classes,
            transform=Resize(input_shape)
        )
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)  
        ##train_model_lr(unet_model, train_loader, val_loader, device, num_classes, epochs=200)
        train_model_lr(unet_model, train_loader, val_loader,result_dir, device, num_classes, epochs=30, learning_rate=5e-4)
    # train_deep_learning()
    def testmodel(imagename):       
        input_shape = (tile_size, tile_size)
        thinsectionpath_image = os.path.join(data_dir,"images",imagename+".tif")
        thinsectionpath_true  = os.path.join( experiment_output_dir, imagename+"_shpmask.tif")
                                             
        model_path = os.path.join( experiment_output_dir,"results/trained_model_20250101_145356.pth")

        output_res_path          =  os.path.join( experiment_output_dir,"testmodels/tiles",imagename)
        predictedtiles_path      =  os.path.join( experiment_output_dir,"testmodels/results/predictedtiles",imagename)
        fullpredicted_path       =  os.path.join( experiment_output_dir,"testmodels/results/predictedtiles", imagename+ "_prediected.tif")

        
        create_tiles(thinsectionpath_image, thinsectionpath_true , tile_size, output_res_path, prefix=imagename)
        model = ResNext101UNet(n_channels=num_channels, n_classes=num_classes)
        model.load_state_dict(torch.load(model_path))
        predict_and_save(model, output_res_path, predictedtiles_path, device, input_shape, num_classes)
        recombine_tiles(thinsectionpath_true,predictedtiles_path, fullpredicted_path)     
    
    imagename = "Chlorite2_after_biotite_Clip"
    # generate_dataset()
    # train_deep_learning()
    testmodel(imagename)

    

    





                
            




            


            
