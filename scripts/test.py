from PIL import Image
from vla import load_vla
import torch
import numpy as np
import pandas as pd
from sim_cogact.adaptive_ensemble import AdaptiveEnsembler

model = load_vla(
        # 'CogACT/CogACT-Base',                   # choose from [CogACT-Small, CogACT-Base, CogACT-Large] or the local path
        '/home/namikosaito/work/exp/own_data_200_exp--image_aug/checkpoints/step-001000-epoch-03-loss=0.0726.pt',
        load_for_training=False, 
        action_model_type='DiT-B',              # choose from ['DiT-S', 'DiT-B', 'DiT-L'] to match the model weight
        future_action_window_size=15,
    )                                 
# about 30G Memory in fp32; 

# (Optional) use "model.vlm = model.vlm.to(torch.bfloat16)" to load vlm in bf16

model.to('cuda:0').eval()

prompt = "pick up the cylinder"               # input your prompt

# Ensemble setup
pred_action_horizon = 10
adaptive_ensemble_alpha = 0.1
ensembler = AdaptiveEnsembler(pred_action_horizon, adaptive_ensemble_alpha)
ensembler.reset()

all_actions = []
for i in range(1, 14):
    frame_path = f"Ep2/frame_{i:05d}.png"
    image = Image.open(frame_path)
    actions, _ = model.predict_action(
        image,
        prompt,
        unnorm_key='own_dataset_200',
        cfg_scale=1.5,
        use_ddim=True,
        num_ddim_steps=10,
    )
    all_actions.append(actions)

# Use AdaptiveEnsembler to ensemble actions
# actions shape: [12, 16, 7] if each actions is [16, 7]
all_actions = np.stack(all_actions, axis=0)  # shape: (12, 16, 7)
ensembled_actions = []
for t in range(all_actions.shape[1]):  # for each step in the action sequence
    # collect the t-th action from each prediction
    step_actions = all_actions[:, t, :]
    ensembled_action = ensembler.ensemble_action(step_actions)
    ensembled_actions.append(ensembled_action)
ensembled_actions = np.stack(ensembled_actions, axis=0)  # shape: (16, 7)

print(ensembled_actions)
np.save("action.npy", ensembled_actions)
pd.DataFrame(ensembled_actions).to_csv("action.csv", index=False, header=False)