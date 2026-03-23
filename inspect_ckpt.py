import torch
ckpt_path = 'waheed_models/unifolm_wma_base.ckpt'
sd = torch.load(ckpt_path, map_location='cpu')
if 'state_dict' in sd:
    sd = sd['state_dict']
print("Number of keys:", len(sd.keys()))
print("First 20 keys:")
for k in list(sd.keys())[:20]:
    print(k)

missing_keys = [
    "agent_action_pos_emb", "agent_state_pos_emb", "cond_pos_emb",
    "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn2.to_k_as.weight"
]

for mk in missing_keys:
    print(f"Is '{mk}' in sd? {'Yes' if mk in sd else 'No'}")
    # also check without 'model.' prefix
    if mk.startswith('model.'):
        mk_no_model = mk[6:]
        print(f"Is '{mk_no_model}' in sd? {'Yes' if mk_no_model in sd else 'No'}")
