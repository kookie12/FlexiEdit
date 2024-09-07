# Example of running MasaCtrl, ProxMasaCtrl, FlexiEdit
from inference_flexiedit import main

if __name__ == "__main__":    

    ### fatty-corgi ###
    main(
        model_path = "../../pretrain_SD_models/CompVis/stable-diffusion-v1-4", #NOTE: please change the path to the model
        out_dir="./outputs/result_flexiedit_test/", 
        source_image_path="./images/fatty-corgi.jpg",
        source_prompt="A corgi is sitting on the floor",
        target_prompt="A corgi is standing on the floor",
        num_inference_steps=50, 
        reinversion_steps=30,
        blended_word=["corgi", "corgi"],        
        cuda_device="cuda:1",
        bbx_start_point = (231, 382),
        bbx_end_point = (456, 455)
    )
     
    ### shepherd ###
    main(
        model_path = "../../pretrain_SD_models/CompVis/stable-diffusion-v1-4",
        out_dir="./outputs/result_flexiedit/",
        source_image_path="./images/shepherd.jpg",
        source_prompt="A dog standing on the ground",
        target_prompt="A dog jumping on the ground", 
        reinversion_steps=20,
        blended_word=["dog", "dog"],
        cuda_device="cuda:0",
        start_point = (199, 356),
        end_point = (484, 466)
    )
    
    ### fatty-corgi ###
    main(
        model_path="../../pretrain_SD_models/stabilityai/stable-diffusion-2-1-base",
        out_dir="./outputs/result_flexiedit/", 
        source_image_path="./images/fatty-corgi.jpg",
        source_prompt="A corgi is sitting on the floor",
        target_prompt="A corgi is standing on the floor",
        reinversion_steps=30,
        blended_word=["corgi", "corgi"],        
        cuda_device="cuda:0",
        bbx_start_point = (231, 382),
        bbx_end_point = (456, 455)
    )
    
    ### teddybear ###
    main(
        model_path="../../pretrain_SD_models/stabilityai/stable-diffusion-2-1-base",
        out_dir="./outputs/result_flexiedit/", 
        source_image_path="./images/teddybear.jpg",
        source_prompt="A teddybear is sitting on the ground",
        target_prompt="A teddybear is running on the ground",
        reinversion_steps=30,
        blended_word= ["teddybear", "teddybear"],
        cuda_device="cuda:0",
        bbx_start_point = (161, 257),
        bbx_end_point = (355, 402)
    )

    
    ### white_horse ###
    main(
        model_path="../../pretrain_SD_models/stabilityai/stable-diffusion-2-1-base",
        out_dir="./outputs/result_flexiedit/", 
        source_image_path="./images/white_horse.jpg",
        source_prompt="A white horse is standing",
        target_prompt="A white horse is running",
        reinversion_steps=30,
        blended_word= ["horse", "horse"],
        cuda_device="cuda:0",
        bbx_start_point = (248, 279),
        bbx_end_point = (385, 400)
    )
    
    ### yellow_cat ###
    main(
        model_path="../../pretrain_SD_models/stabilityai/stable-diffusion-2-1-base",
        out_dir="./outputs/result_flexiedit/", 
        source_image_path="./images/yellow_cat.jpg",
        source_prompt="A cat",
        target_prompt="A cat with side view",
        reinversion_steps=40,
        blended_word= ["cat", "cat"],
        cuda_device="cuda:0",
        bbx_start_point = (119, 6),
        bbx_end_point = (261, 158)
    )
    