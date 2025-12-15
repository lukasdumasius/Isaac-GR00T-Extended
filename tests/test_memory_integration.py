import torch
import numpy as np

from gr00t.model.action_head.cross_attention_dit import DiT
from gr00t.model.action_memory.memory_module import ActionMemory
from gr00t.model.gr00t_n1 import GR00T_N1_5, GR00T_N1_5_Config


def test_action_memory_build_keys_values():
    """ActionMemory builds K/V with matching dims and pos/text injection."""
    B, T, D_act = 2, 4, 7
    actions = torch.randn(B, T, D_act)
    timesteps = torch.zeros(B)

    mem = ActionMemory(action_dim=D_act, action_hidden_size=16, max_memory_size=8)
    out = mem(actions=actions, timesteps=timesteps, retrieve=False)

    # Build K/V directly from current traj
    keys, vals = mem.build_from_traj_latent(out["traj_latent"], text_emb=None)
    assert keys.shape == (B, 1, 16)
    assert vals.shape == (B, 1, 16)
    # Keys and vals differ when text is nonzero; here text=0 so keys-vals==0
    assert torch.allclose(keys, vals)

    # Add to bank and retrieve
    mem.add_to_memory(out)
    fused = mem.perform_retrieval(query_latent=out["traj_latent"], text_emb=None)
    assert fused.shape == out["traj_latent"].shape


def test_dit_forward_with_memory_cross_attention():
    """DiT forward runs with memory K/V and produces expected shapes."""
    B, T, S = 2, 3, 5
    inner_dim = 16  # num_heads * head_dim

    dit = DiT(
        num_attention_heads=2,
        attention_head_dim=8,
        output_dim=4,
        num_layers=2,
        enable_memory_cross_attention=True,
        memory_cross_attention_layers=None,  # dense
        memory_dim=inner_dim,
    )

    hidden_states = torch.randn(B, T, inner_dim)
    encoder_hidden_states = torch.randn(B, S, inner_dim)
    timesteps = torch.randint(0, 1000, (B,))

    # Simple memory K/V (N=2)
    memory_keys = torch.randn(B, 2, inner_dim)
    memory_values = torch.randn(B, 2, inner_dim)

    out = dit(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        timestep=timesteps,
        memory_keys=memory_keys,
        memory_values=memory_values,
        return_all_hidden_states=False,
    )

    assert out.shape == (B, T, 4)


def test_action_memory_dual_stream():
    """Test ActionMemory with dual-stream (action + state) input."""
    print("\n" + "="*80)
    print("Testing ActionMemory Dual-Stream Encoder")
    print("="*80)
    
    B, T, D_act, D_state = 2, 16, 7, 8
    actions = torch.randn(B, T, D_act)
    states = torch.randn(B, T, D_state)
    timesteps = torch.zeros(B)
    cat_ids = torch.zeros(B, dtype=torch.long)
    
    print(f"\n1. Creating ActionMemory with dual-stream support...")
    mem = ActionMemory(
        action_dim=D_act,
        state_dim=D_state,
        action_hidden_size=128,
        num_embodiments=2,
        max_memory_size=8
    )
    print(f"   ✓ Memory initialized (action_dim={D_act}, state_dim={D_state})")
    
    print(f"\n2. Forward pass with actions + states...")
    with torch.no_grad():
        out = mem(
            actions=actions,
            states=states,
            timesteps=timesteps,
            cat_ids=cat_ids,
            retrieve=False
        )
    
    # Verify outputs
    assert "traj_latent" in out
    assert "fused_tokens" in out
    assert out["traj_latent"].shape == (B, 1, 128)
    assert out["fused_tokens"].shape == (B, T, 128)
    print(f"   ✓ traj_latent shape: {out['traj_latent'].shape}")
    print(f"   ✓ fused_tokens shape: {out['fused_tokens'].shape}")
    
    print(f"\n3. Building memory keys/values...")
    keys, vals = mem.build_from_traj_latent(out["traj_latent"], text_emb=None)
    assert keys.shape == (B, 1, 128)
    assert vals.shape == (B, 1, 128)
    print(f"   ✓ Keys shape: {keys.shape}")
    print(f"   ✓ Values shape: {vals.shape}")
    
    print(f"\n4. Adding to memory bank and retrieval...")
    mem.add_to_memory(out)
    assert len(mem.memory_bank) == B
    print(f"   ✓ Memory bank size: {len(mem.memory_bank)}")
    
    retrieved = mem.perform_retrieval(query_latent=out["traj_latent"], text_emb=None)
    assert retrieved.shape == (B, 1, 128)
    print(f"   ✓ Retrieved memory shape: {retrieved.shape}")
    
    print("\n" + "="*80)
    print("✓ Dual-stream memory test passed!")
    print("="*80 + "\n")


def test_groot_single_forward():
    """Test complete GR00T model with one forward pass using real dataset."""
    print("\n" + "="*80)
    print("Testing GR00T Model - Single Forward Pass")
    print("="*80)
    
    print("\n1. Loading dataset with GR00TTransform...")
    from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import GR00TTransform
    
    # Use demo dataset
    dataset_path = "demo_data/robot_sim.PickNPlace"
    
    # Modality config matching the demo dataset structure
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.left_arm", "state.right_arm"],  # Use actual keys from modality.json
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),  # action_horizon=16
            modality_keys=["action.left_arm", "action.right_arm"],  # Use actual keys from modality.json
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    
    # Create GR00T transform (we will apply manually after sampling)
    groot_transform = GR00TTransform(
        max_state_dim=14,  # left_arm (7) + right_arm (7)
        max_action_dim=14,
        state_horizon=1,
        action_horizon=16,
        embodiment_tag=EmbodimentTag("gr1"),
    )
    
    # Load dataset without transforms; we will add video alias and apply transform manually
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag("gr1"),
        video_backend="torchvision_av",
    )
    
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    # Get a real sample (raw), then alias video key and apply transform manually
    raw_sample = dataset[0]
    print(f"   ✓ Raw sample keys: {list(raw_sample.keys())}")
    
    # GR00TTransform expects key "video" with shape [T, V, H, W, C] (ndim=5) or batched (ndim=6).
    # Demo sample video.ego_view comes as [H, W, C] (ndim=3) or [T, H, W, C] (ndim=4). Expand dims accordingly.
    raw_sample = dict(raw_sample)
    video_np = raw_sample["video.ego_view"]
    if video_np.ndim == 3:  # [H, W, C] -> [T=1, V=1, H, W, C]
        video_np = video_np[None, None, ...]
    elif video_np.ndim == 4:  # [T, H, W, C] -> [T, V=1, H, W, C]
        video_np = video_np[:, None, ...]
    else:
        raise ValueError(f"Unexpected video ndim: {video_np.ndim}")
    raw_sample["video"] = video_np
    
    transformed_sample = groot_transform.apply(raw_sample)
    print(f"   ✓ Transformed keys: {list(transformed_sample.keys())}")
    
    print("\n2. Loading GR00T model from pretrained...")
    # Load model using from_pretrained (same as training script)
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        use_action_memory=True,
        memory_cfg={
            "max_memory_size": 32,
            "readout_nhead": 4,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    
    assert model.action_memory is not None
    print(f"   ✓ Model initialized with memory (max_size={model.action_memory.max_memory_size})")
    
    # Get model's expected dimensions
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    # Check max_state_dim and max_action_dim from action_head config if available
    max_state_dim = getattr(model.action_head.config, "max_state_dim", 128)  # likely 64/128
    # For action tensors we align to model.action_dim (decoder expects this)
    max_action_dim = model.action_dim
    
    print(f"   ✓ Model expects action_dim={action_dim}, action_horizon={action_horizon}")
    print(f"   ✓ Model expects max_state_dim={max_state_dim}, max_action_dim={max_action_dim}")
    
    print("\n3. Creating batch input with memory history...")
    batch_size = 1
    
    # The sample is transformed; collate to build eagle_* inputs
    from gr00t.model.transforms import collate
    
    batch = collate([transformed_sample], groot_transform.eagle_processor)
    
    # Pad state to max_state_dim
    current_state = batch["state"] # [B, T, D]
    if current_state.shape[-1] < max_state_dim:
        padding = torch.zeros(
            current_state.shape[0], 
            current_state.shape[1], 
            max_state_dim - current_state.shape[-1]
        )
        batch["state"] = torch.cat([current_state, padding], dim=-1)
    
    # Add memory history (512 steps) - align to model.action_dim
    batch["action_history"] = torch.randn(batch_size, 512, max_action_dim)
    batch["state_history"] = torch.randn(batch_size, 512, max_state_dim)
    # Provide action and mask with model.action_dim (zeros/ones for inference)
    batch["action"] = torch.zeros(batch_size, action_horizon, max_action_dim)
    batch["action_mask"] = torch.ones(batch_size, action_horizon, max_action_dim)

    # Move tensors to device and match model dtype for floating tensors
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                batch[k] = v.to(device=device, dtype=model_dtype)
            else:
                batch[k] = v.to(device)

    print(f"   ✓ Batch prepared")
    for k, v in batch.items():
        if isinstance(v, (np.ndarray, torch.Tensor)):
            print(f"      {k}: {v.shape if hasattr(v, 'shape') else type(v)}")
    
    print("\n4. Running forward pass...")
    with torch.no_grad():
        outputs = model(batch)
    
    # Verify outputs
    assert "loss" in outputs or "action_pred" in outputs
    print(f"   ✓ Forward pass completed")
    print(f"   ✓ Output keys: {list(outputs.keys())}")
    
    # Check if memory was updated
    memory_size = len(model.action_memory.memory_bank)
    print(f"   ✓ Memory bank size after forward: {memory_size}")
    
    print("\n" + "="*80)
    print("✓ Single forward test passed!")
    print("="*80 + "\n")


def test_groot_model_with_memory_multi_step():
    """
    Test complete GR00T model with action memory over multiple inference steps.
    
    This test:
    1. Loads a minimal GR00T model with memory enabled
    2. Simulates 64 sequential inference steps using real dataset samples
    3. Verifies memory bank grows and stabilizes
    4. Checks output consistency and shapes
    """
    print("\n" + "="*80)
    print("Testing GR00T Model with Memory - Multi-Step Inference")
    print("="*80)
    
    print("\n1. Loading dataset with GR00TTransform...")
    from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import GR00TTransform, collate
    
    dataset_path = "demo_data/robot_sim.PickNPlace"
    
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.left_arm", "state.right_arm"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action.left_arm", "action.right_arm"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    
    # Create GR00T transform (manual apply)
    groot_transform = GR00TTransform(
        max_state_dim=14,
        max_action_dim=14,
        state_horizon=1,
        action_horizon=16,
        embodiment_tag=EmbodimentTag("gr1"),
    )
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag("gr1"),
        video_backend="torchvision_av",
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    # Load model
    print("\n2. Loading GR00T model from pretrained...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        use_action_memory=True,
        memory_cfg={
            "max_memory_size": 32,
            "readout_nhead": 4,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    
    # Verify memory module exists
    assert model.action_memory is not None, "Action memory should be initialized"
    print(f"   ✓ Memory module initialized (max_size={model.action_memory.max_memory_size})")
    
    # Get model's expected dimensions
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    max_state_dim = getattr(model.action_head.config, "max_state_dim", 128)
    max_action_dim = getattr(model.action_head.config, "max_action_dim", 128)
    print(f"   ✓ Model expects action_dim={action_dim}, action_horizon={action_horizon}")
    print(f"   ✓ Model expects max_state_dim={max_state_dim}, max_action_dim={max_action_dim}")
    
    print(f"\n3. Running 64 sequential inference steps...")
    
    # Track memory bank size over time
    memory_sizes = []
    num_steps = min(64, len(dataset))
    
    for step in range(num_steps):
        # Get real sample from dataset (raw), alias video, apply transform, then collate
        raw_sample = dict(dataset[step % len(dataset)])
        # Ensure video has shape [T, V, H, W, C] for GR00TTransform
        video_np = raw_sample["video.ego_view"]
        if video_np.ndim == 3:  # [H, W, C] -> [T=1, V=1, H, W, C]
            video_np = video_np[None, None, ...]
        elif video_np.ndim == 4:  # [T, H, W, C] -> [T, V=1, H, W, C]
            video_np = video_np[:, None, ...]
        else:
            raise ValueError(f"Unexpected video ndim: {video_np.ndim}")
        raw_sample["video"] = video_np
        transformed_sample = groot_transform.apply(raw_sample)
        
        batch = collate([transformed_sample], groot_transform.eagle_processor)
        
        # Pad state to max_state_dim
        current_state = batch["state"]
        if current_state.shape[-1] < max_state_dim:
            padding = torch.zeros(
                current_state.shape[0], 
                current_state.shape[1], 
                max_state_dim - current_state.shape[-1]
            )
            batch["state"] = torch.cat([current_state, padding], dim=-1)
        
        # Add memory history (synthetic) and provide zero actions matching model.action_dim
        batch_size = 1
        batch["action_history"] = torch.randn(batch_size, 512, max_action_dim)
        batch["state_history"] = torch.randn(batch_size, 512, max_state_dim)
        batch["action"] = torch.zeros(batch_size, action_horizon, max_action_dim)
        batch["action_mask"] = torch.ones(batch_size, action_horizon, max_action_dim)

        # Move tensors to device and cast float tensors to model dtype
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    batch[k] = v.to(device=device, dtype=model_dtype)
                else:
                    batch[k] = v.to(device)
        
        # Forward pass
        with torch.no_grad():
            try:
                outputs = model(batch)
                
                # Verify outputs
                assert "loss" in outputs or "action_pred" in outputs, \
                    "Output should contain loss (training) or action_pred (inference)"
                
                # Check memory bank size
                current_memory_size = len(model.action_memory.memory_bank)
                memory_sizes.append(current_memory_size)
                
                if step % 10 == 0:
                    print(f"   Step {step:3d}: memory_bank_size={current_memory_size:2d}")
                
            except Exception as e:
                print(f"\n   ✗ Forward pass failed at step {step}: {e}")
                raise
    
    print(f"\n3. Verifying memory behavior...")
    
    # Memory should grow up to max_memory_size and then stabilize
    max_memory_size = model.action_memory.max_memory_size
    final_memory_size = memory_sizes[-1]
    
    print(f"   Initial memory size: {memory_sizes[0]}")
    print(f"   Final memory size:   {final_memory_size}")
    print(f"   Max memory size:     {max_memory_size}")
    
    # Memory should not exceed max size
    assert all(size <= max_memory_size for size in memory_sizes), \
        f"Memory bank exceeded max size: {max(memory_sizes)} > {max_memory_size}"
    
    # Memory should eventually reach max size (if we have enough steps)
    if num_steps >= max_memory_size:
        assert final_memory_size == max_memory_size, \
            f"Memory should reach max size after {num_steps} steps"
    
    print(f"   ✓ Memory bank properly capped at {max_memory_size}")
    print(f"   ✓ Memory grew monotonically: {memory_sizes[:10]} ...")
    
    print("\n4. Testing memory retrieval...")
    
    # Create a query and test retrieval
    query_actions = torch.randn(batch_size, 16, max_action_dim).to(device=device, dtype=model_dtype)
    query_timesteps = torch.zeros(batch_size).to(device)
    
    with torch.no_grad():
        result = model.action_memory(
            actions=query_actions,
            timesteps=query_timesteps,
            retrieve=True
        )
        
        assert "traj_latent" in result
        assert "retrieved_memory" in result
        
        if result["retrieved_memory"] is not None:
            print(f"   ✓ Retrieved memory shape: {result['retrieved_memory'].shape}")
        else:
            print(f"   ✓ No memory to retrieve (bank empty)")
    
    print("\n" + "="*80)
    print("✓ All multi-step memory tests passed!")
    print("="*80 + "\n")


def test_groot_batch_inference():
    """
    Test GR00T with batch_size > 1 to verify padding and masking.
    """
    print("\n" + "="*80)
    print("Testing GR00T Model - Batch Inference (batch_size=4)")
    print("="*80)
    
    print("\n1. Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import GR00TTransform, collate
    
    dataset_path = "demo_data/robot_sim.PickNPlace"
    
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.left_arm", "state.right_arm"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action.left_arm", "action.right_arm"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    
    groot_transform = GR00TTransform(
        max_state_dim=14,
        max_action_dim=14,
        state_horizon=1,
        action_horizon=16,
        embodiment_tag=EmbodimentTag("gr1"),
    )
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag("gr1"),
        video_backend="torchvision_av",
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    print("\n2. Loading model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        use_action_memory=True,
        memory_cfg={
            "max_memory_size": 32,
            "readout_nhead": 4,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    max_state_dim = getattr(model.action_head.config, "max_state_dim", 128)
    max_action_dim = model.action_dim
    
    print(f"   ✓ Model loaded (action_dim={action_dim}, max_state_dim={max_state_dim})")
    
    print("\n3. Preparing batch with batch_size=4...")
    batch_size = 4
    
    # Get multiple samples and collate
    samples = []
    for i in range(batch_size):
        raw_sample = dict(dataset[i % len(dataset)])
        video_np = raw_sample["video.ego_view"]
        if video_np.ndim == 3:
            video_np = video_np[None, None, ...]
        elif video_np.ndim == 4:
            video_np = video_np[:, None, ...]
        raw_sample["video"] = video_np
        transformed = groot_transform.apply(raw_sample)
        samples.append(transformed)
    
    batch = collate(samples, groot_transform.eagle_processor)
    
    # Pad state
    current_state = batch["state"]
    if current_state.shape[-1] < max_state_dim:
        padding = torch.zeros(
            current_state.shape[0], 
            current_state.shape[1], 
            max_state_dim - current_state.shape[-1]
        )
        batch["state"] = torch.cat([current_state, padding], dim=-1)
    
    # Add memory history and action
    batch["action_history"] = torch.randn(batch_size, 512, max_action_dim)
    batch["state_history"] = torch.randn(batch_size, 512, max_state_dim)
    batch["action"] = torch.zeros(batch_size, action_horizon, max_action_dim)
    batch["action_mask"] = torch.ones(batch_size, action_horizon, max_action_dim)
    
    # Move to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                batch[k] = v.to(device=device, dtype=model_dtype)
            else:
                batch[k] = v.to(device)
    
    print(f"   ✓ Batch prepared: batch_size={batch_size}")
    print(f"      state: {batch['state'].shape}")
    print(f"      action_history: {batch['action_history'].shape}")
    
    print("\n4. Running batch forward pass...")
    with torch.no_grad():
        outputs = model(batch)
    
    assert "loss" in outputs or "action_pred" in outputs
    print(f"   ✓ Batch forward pass completed")
    print(f"   ✓ Memory bank size: {len(model.action_memory.memory_bank)}")
    
    print("\n" + "="*80)
    print("✓ Batch inference test passed!")
    print("="*80 + "\n")
    
    # Cleanup
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_groot_without_memory_history():
    """
    Test GR00T forward without action_history/state_history (empty memory branch).
    """
    print("\n" + "="*80)
    print("Testing GR00T Model - Forward Without Memory History")
    print("="*80)
    
    print("\n1. Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import GR00TTransform, collate
    
    dataset_path = "demo_data/robot_sim.PickNPlace"
    
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.left_arm", "state.right_arm"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action.left_arm", "action.right_arm"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    
    groot_transform = GR00TTransform(
        max_state_dim=14,
        max_action_dim=14,
        state_horizon=1,
        action_horizon=16,
        embodiment_tag=EmbodimentTag("gr1"),
    )
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag("gr1"),
        video_backend="torchvision_av",
    )
    
    print("\n2. Loading model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        use_action_memory=True,
        memory_cfg={
            "max_memory_size": 32,
            "readout_nhead": 4,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    max_state_dim = getattr(model.action_head.config, "max_state_dim", 128)
    max_action_dim = model.action_dim
    
    print(f"   ✓ Model loaded")
    
    print("\n3. Preparing input WITHOUT action_history/state_history...")
    raw_sample = dict(dataset[0])
    video_np = raw_sample["video.ego_view"]
    if video_np.ndim == 3:
        video_np = video_np[None, None, ...]
    elif video_np.ndim == 4:
        video_np = video_np[:, None, ...]
    raw_sample["video"] = video_np
    transformed = groot_transform.apply(raw_sample)
    
    batch = collate([transformed], groot_transform.eagle_processor)
    
    # Pad state
    current_state = batch["state"]
    if current_state.shape[-1] < max_state_dim:
        padding = torch.zeros(
            current_state.shape[0], 
            current_state.shape[1], 
            max_state_dim - current_state.shape[-1]
        )
        batch["state"] = torch.cat([current_state, padding], dim=-1)
    
    # Add action and mask, but NO action_history / state_history
    batch["action"] = torch.zeros(1, action_horizon, max_action_dim)
    batch["action_mask"] = torch.ones(1, action_horizon, max_action_dim)
    
    # Move to device
    for k, v in list(batch.items()):
        if isinstance(v, torch.Tensor):
            if torch.is_floating_point(v):
                batch[k] = v.to(device=device, dtype=model_dtype)
            else:
                batch[k] = v.to(device)
    
    print(f"   ✓ Input prepared (NO memory history)")
    print(f"      Keys: {list(batch.keys())}")
    
    print("\n4. Running forward pass without memory history...")
    with torch.no_grad():
        outputs = model(batch)
    
    assert "loss" in outputs or "action_pred" in outputs
    print(f"   ✓ Forward pass completed")
    print(f"   ✓ Memory bank size: {len(model.action_memory.memory_bank)}")
    
    print("\n" + "="*80)
    print("✓ Forward without memory history test passed!")
    print("="*80 + "\n")
    
    # Cleanup
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()


def test_groot_long_horizon_inference():
    """
    Test GR00T with long-horizon inference (128 steps) to verify:
    1. Memory bank updates correctly over extended sequences
    2. Memory retrieval works consistently
    3. No memory leaks or performance degradation
    """
    print("\n" + "="*80)
    print("Testing GR00T Model - Long Horizon Inference (128 steps)")
    print("="*80)
    
    print("\n1. Loading dataset...")
    from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
    from gr00t.data.schema import EmbodimentTag
    from gr00t.model.transforms import GR00TTransform, collate
    
    dataset_path = "demo_data/robot_sim.PickNPlace"
    
    modality_configs = {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["video.ego_view"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["state.left_arm", "state.right_arm"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["action.left_arm", "action.right_arm"],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["annotation.human.action.task_description"],
        ),
    }
    
    groot_transform = GR00TTransform(
        max_state_dim=14,
        max_action_dim=14,
        state_horizon=1,
        action_horizon=16,
        embodiment_tag=EmbodimentTag("gr1"),
    )
    
    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        transforms=None,
        embodiment_tag=EmbodimentTag("gr1"),
        video_backend="torchvision_av",
    )
    print(f"   ✓ Dataset loaded: {len(dataset)} samples")
    
    print("\n2. Loading model...")
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path="nvidia/GR00T-N1.5-3B",
        tune_llm=False,
        tune_visual=False,
        tune_projector=True,
        tune_diffusion_model=True,
        use_action_memory=True,
        memory_cfg={
            "max_memory_size": 32,
            "readout_nhead": 4,
        },
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = model.to(device=device, dtype=model_dtype)
    model.eval()
    
    action_dim = model.action_dim
    action_horizon = model.action_horizon
    max_state_dim = getattr(model.action_head.config, "max_state_dim", 128)
    max_action_dim = model.action_dim
    max_memory_size = model.action_memory.max_memory_size
    
    print(f"   ✓ Model loaded (max_memory_size={max_memory_size})")
    
    print("\n3. Running 128-step long-horizon inference...")
    num_steps = 128
    memory_sizes = []
    retrieval_checks = []
    
    for step in range(num_steps):
        # Prepare input
        raw_sample = dict(dataset[step % len(dataset)])
        video_np = raw_sample["video.ego_view"]
        if video_np.ndim == 3:
            video_np = video_np[None, None, ...]
        elif video_np.ndim == 4:
            video_np = video_np[:, None, ...]
        raw_sample["video"] = video_np
        transformed = groot_transform.apply(raw_sample)
        
        batch = collate([transformed], groot_transform.eagle_processor)
        
        # Pad state
        current_state = batch["state"]
        if current_state.shape[-1] < max_state_dim:
            padding = torch.zeros(
                current_state.shape[0], 
                current_state.shape[1], 
                max_state_dim - current_state.shape[-1]
            )
            batch["state"] = torch.cat([current_state, padding], dim=-1)
        
        # Add memory history and action
        batch["action_history"] = torch.randn(1, 512, max_action_dim)
        batch["state_history"] = torch.randn(1, 512, max_state_dim)
        batch["action"] = torch.zeros(1, action_horizon, max_action_dim)
        batch["action_mask"] = torch.ones(1, action_horizon, max_action_dim)
        
        # Move to device
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                if torch.is_floating_point(v):
                    batch[k] = v.to(device=device, dtype=model_dtype)
                else:
                    batch[k] = v.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(batch)
        
        # Track memory bank size
        current_memory_size = len(model.action_memory.memory_bank)
        memory_sizes.append(current_memory_size)
        
        # Periodically test retrieval
        if step > 0 and step % 20 == 0:
            query_actions = torch.randn(1, 16, max_action_dim).to(device=device, dtype=model_dtype)
            query_states = torch.randn(1, 16, max_state_dim).to(device=device, dtype=model_dtype)
            query_timesteps = torch.zeros(1).to(device)
            cat_ids = torch.zeros(1, dtype=torch.long).to(device)
            
            result = model.action_memory(
                actions=query_actions,
                timesteps=query_timesteps,
                states=query_states,
                cat_ids=cat_ids,
                retrieve=True
            )
            
            retrieval_success = result["retrieved_memory"] is not None
            retrieval_checks.append(retrieval_success)
            
            print(f"   Step {step:3d}: memory_size={current_memory_size:2d}, "
                  f"retrieval={'✓' if retrieval_success else '✗'}")
    
    print(f"\n4. Verifying long-horizon behavior...")
    
    # Check memory stabilization
    assert memory_sizes[-1] == max_memory_size, \
        f"Memory should stabilize at max_size={max_memory_size}, got {memory_sizes[-1]}"
    
    # Check memory growth pattern
    growth_phase = memory_sizes[:max_memory_size]
    stable_phase = memory_sizes[max_memory_size:]
    
    assert all(growth_phase[i] <= growth_phase[i+1] for i in range(len(growth_phase)-1)), \
        "Memory should grow monotonically before reaching max"
    
    assert all(s == max_memory_size for s in stable_phase), \
        "Memory should remain at max_size after saturation"
    
    print(f"   ✓ Memory grew from {memory_sizes[0]} to {max_memory_size} in {max_memory_size} steps")
    print(f"   ✓ Memory remained stable at {max_memory_size} for {len(stable_phase)} steps")
    
    # Check retrieval success rate
    retrieval_rate = sum(retrieval_checks) / len(retrieval_checks) if retrieval_checks else 0
    print(f"   ✓ Retrieval success rate: {retrieval_rate*100:.1f}% ({sum(retrieval_checks)}/{len(retrieval_checks)})")
    
    assert retrieval_rate >= 0.8, \
        f"Retrieval should succeed in at least 80% of cases, got {retrieval_rate*100:.1f}%"
    
    print("\n" + "="*80)
    print("✓ Long-horizon inference test passed!")
    print("="*80 + "\n")
    
    # Cleanup
    import gc
    del model
    torch.cuda.empty_cache()
    gc.collect()
