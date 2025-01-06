import os 
from datapreparation.generate_Dataset import generate_tif
from datapreparation.generate_mask2 import generate_mask
from machinelearning.classifypixel3 import classifypixel
from datapreparation.arctrain  import create_shapefile_from_mask

import os
from pathlib import Path

def check_matching_files(base_directory):
    # Convert to Path object
    base_path = Path(base_directory)
    
    # Define specific subfolder paths
    images_path = Path.joinpath(base_path,'images')
    shps_path = Path.joinpath(base_path , 'shapes')
    
    # Dictionaries to store files
    tif_files = {}
    shp_files = {}
    
    # Get all TIF files
    if images_path.exists():
        for file in images_path.rglob('*.tif'):
            tif_files[file.stem] = str(file)
        for file in images_path.rglob('*.tiff'):
            tif_files[file.stem] = str(file)
    
    # Get all shapefiles
    if shps_path.exists():
        for file in shps_path.rglob('vec_*.shp'):
            clean_name = file.stem.replace('Vec_', '')
            shp_files[clean_name] = str(file)
    
    # Check for matches and mismatches
    tifs_without_shp = []
    shps_without_tif = []
    matching_pairs = []
    
    # Check TIFs that don't have matching shapefiles
    for tif_name in tif_files:
        if tif_name in shp_files:
            matching_pairs.append((tif_files[tif_name], shp_files[tif_name]))
        else:
            tifs_without_shp.append(tif_files[tif_name])
    
    # Check shapefiles that don't have matching TIFs
    for shp_name in shp_files:
        if shp_name not in tif_files:
            shps_without_tif.append(shp_files[shp_name])
    
    # Print results
    print("Matching pairs (TIF and SHP):")
    for tif, shp in matching_pairs:
        print(f"TIF: {tif}\nSHP: {shp}\n")

    print("\nTIF files without matching shapefiles:")
    for tif in tifs_without_shp:
        print(tif)

    print("\nShapefiles without matching TIF files:")
    for shp in shps_without_tif:
        print(shp)

    # Print summary
    print(f"\nSummary:")
    print(f"Total matching pairs: {len(matching_pairs)}")
    print(f"TIFs without shapefiles: {len(tifs_without_shp)}")
    print(f"Shapefiles without TIFs: {len(shps_without_tif)}")
    
    
    return matching_pairs, tifs_without_shp, shps_without_tif




def main1():
    data_dir    = r'D:\narssprojects\thensections\6bands\data\group1'      
    
    tif_file   = os.path.join( data_dir, 'images/Biotite_Gr_Clip.tif')
    shapefile  = os.path.join(data_dir,'shapes/Vec_Biotite_Gr_Clip.shp')    

    outputdir =  r"D:\narssprojects\thensections\output"
    experdir  =  "ML01"
    experPath =  os.path.join(outputdir,experdir)
    os.makedirs(experPath, exist_ok=True)


    mask_file  = os.path.join( experPath,'Biotite_Gr_Clip_mask.tif')

    arcshapefile  = os.path.join(experPath,'arcExport_Output.shp')  

    classification_file=  os.path.join( experPath,'output_classification_mask_rf4') 
   
    generate_mask(shapefile, tif_file, mask_file)
    classifypixel(tif_file,mask_file,classification_file,train =True,classifier_prefix='rf')
    create_shapefile_from_mask(mask_file,arcshapefile)


def main2():
    # Example usage
    directory = r"D:\narssprojects\thensections\6bands\data\group1"  # Replace with your base directory path
    outputdir =  r"D:\narssprojects\thensections\output"
    experdir  =  "ML01"
    experPath =  os.path.join(outputdir,experdir)
    os.makedirs(experPath, exist_ok=True)


    matching_pairs, tifs_without_shp, shps_without_tif = check_matching_files(directory)
    for tif_file, shapefile in matching_pairs:
        print(f"TIF: {tif_file}\nSHP: {shapefile}\n")
        filename_no_ext = os.path.splitext(os.path.basename(tif_file))[0]  # 'Biotite2_Gr_Clip'

        mask_file  = os.path.join( experPath, filename_no_ext + '_mask.tif')
        arcshapefile  = os.path.join(experPath,'arcExport_'+filename_no_ext+'.shp') 

        classification_file=  os.path.join( experPath, filename_no_ext +  '_predicted_mask_rf') 
    
        generate_mask(shapefile, tif_file, mask_file)
        classifypixel(tif_file,mask_file,classification_file,train =True,classifier_prefix='rf')
        create_shapefile_from_mask(mask_file,arcshapefile)
        
    




if __name__ == "__main__":   
    main2()
    

