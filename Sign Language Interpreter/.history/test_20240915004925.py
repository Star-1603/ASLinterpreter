import os



file_name = "asl_model.h5'"  



if os.path.exists(file_name):  

    print(f"File '{file_name}' exists in the current directory.") 

else:

    print(f"File '{file_name}' does not exist in the current directory.") 