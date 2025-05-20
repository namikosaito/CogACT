from PIL import Image
from vla import load_vla
import torch

model = load_vla(
        # 'CogACT/CogACT-Base',                   # choose from [CogACT-Small, CogACT-Base, CogACT-Large] or the local path
        '/home/namikosaito/work/exp/test_exp--image_aug/checkpoints/step-000250-epoch-08-loss=0.3239.pt',
        load_for_training=False, 
        action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
        future_action_window_size=15,
    )                                 
# about 30G Memory in fp32; 

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

image: Image.Image = Image.open("apple_image.png")     
prompt = "move sponge near apple"               # input your prompt

# Predict Action (7-DoF; un-normalize for RT-1 google robot data, i.e., fractal20220817_data)
actions, _ = model.predict_action(
            image,
            prompt,
            # unnorm_key='fractal20220817_data',  # input your unnorm_key of the dataset
            unnorm_key='own_dataset',
            cfg_scale = 1.5,                    # cfg from 1.5 to 7 also performs well
            use_ddim = True,                    # use DDIM sampling
            num_ddim_steps = 10,                # number of steps for DDIM sampling
        )

    # results in 7-DoF actions of 16 steps with shape [16, 7]
print(actions)