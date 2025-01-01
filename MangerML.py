import os 
from datapreparation.generate_Dataset import generate_tif
from datapreparation.generate_mask2 import generate_mask
from machinelearning.classifypixel3 import classifypixel
from datapreparation.arctrain  import create_shapefile_from_mask


if __name__ == "__main__":   
    data_dir    = r'F:\Senaa\PROJECT\thensections\6bands\data\group1'      
    tif_file   = os.path.join( data_dir, 'images\Biotite_Gr_Clip.tif')
    shapefile  = os.path.join(data_dir,'shapes\Vec_Biotite_Gr_Clip.shp')    

    outputdir =  r"F:\Senaa\PROJECT\thensections\output"
    experdir  =  "ML01"
    experPath =  os.path.join(outputdir,experdir)
    os.makedirs(experPath, exist_ok=True)


    mask_file  = os.path.join( experPath,'Biotite_Gr_Clip_mask.tif')

    arcshapefile  = os.path.join(experPath,'arcExport_Output.shp')  

    classification_file=  os.path.join( experPath,'output_classification_mask_rf4') 
   
    generate_mask(shapefile, tif_file, mask_file)
    classifypixel(tif_file,mask_file,classification_file,train =True,classifier_prefix='rf')
    create_shapefile_from_mask(mask_file,arcshapefile)


