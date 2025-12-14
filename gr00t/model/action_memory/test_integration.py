"""
Simple integration test for action memory module.
Run this to verify the implementation works correctly.
"""

import torch
from gr00t.model.action_memory.memory_cross_attention import (
    MemoryCrossAttention,
    MemoryBank,
    TrajectoryEncoder,
)
from gr00t.model.action_memory.memory_training_utils import (
    ActionMemoryTrainingWrapper,
    ActionMemoryInferenceManager,
)


def test_trajectory_encoder():
    """Test trajectory encoder."""
    print("Testing TrajectoryEncoder...")
    
    encoder = TrajectoryEncoder(
        action_dim=7,
        hidden_dim=256,
        output_dim=512,
        window_size=10,
        encoder_type="mlp",
    )
    
    # Test input
    actions = torch.randn(2, 10, 7)  # (B, window_size, action_dim)
    
    # Encode
    latent = encoder(actions)
    
    assert latent.shape == (2, 1, 512), f"Expected (2, 1, 512), got {latent.shape}"
    print("✓ TrajectoryEncoder passed")


def test_memory_bank():
    """Test memory bank."""
    print("\nTesting MemoryBank...")
    
    bank = MemoryBank(max_capacity=5, eviction_strategy="fifo")
    
    # Add memories
    for i in range(7):
        mem = torch.randn(1, 1, 512)
        bank.add(mem)
    
    # Should evict 2 oldest
    assert len(bank) == 5, f"Expected 5 memories, got {len(bank)}"
    
    # Get memory
    keys, values = bank.get_memory()
    assert keys.shape == (1, 5, 512), f"Expected (1, 5, 512), got {keys.shape}"
    assert values.shape == (1, 5, 512), f"Expected (1, 5, 512), got {values.shape}"
    
    print("✓ MemoryBank passed")


def test_memory_cross_attention():
    """Test memory cross-attention."""
    print("\nTesting MemoryCrossAttention...")
    
    attn = MemoryCrossAttention(
        query_dim=512,
        memory_dim=512,
        num_heads=8,
        dropout=0.0,
    )
    
    # Test inputs
    query = torch.randn(2, 4, 512)  # (B, T, D)
    memory_keys = torch.randn(2, 10, 512)  # (B, N, D)
    memory_values = torch.randn(2, 10, 512)  # (B, N, D)
    
    # Apply attention
    output = attn(query, memory_keys, memory_values)
    
    assert output.shape == (2, 4, 512), f"Expected (2, 4, 512), got {output.shape}"
    print("✓ MemoryCrossAttention passed")


def test_training_wrapper():
    """Test training wrapper."""
    print("\nTesting ActionMemoryTrainingWrapper...")
    
    wrapper = ActionMemoryTrainingWrapper(
        action_dim=7,
        memory_dim=512,
        trajectory_window=10,
        max_memory_capacity=100,
        device="cpu",
    )
    
    # Create batch of actions
    batch_actions = torch.randn(4, 50, 7)  # (B, T_total, D_action)
    
    # Build memory
    memory_keys, memory_values = wrapper.build_per_sample_memory(batch_actions)
    
    # Check shapes
    assert memory_keys.shape[0] == 4, f"Expected batch size 4, got {memory_keys.shape[0]}"
    assert memory_keys.shape[2] == 512, f"Expected memory_dim 512, got {memory_keys.shape[2]}"
    assert memory_keys.shape == memory_values.shape, "Keys and values should have same shape"
    
    print(f"  Memory shape: {memory_keys.shape}")
    print("✓ ActionMemoryTrainingWrapper passed")


def test_inference_manager():
    """Test inference manager."""
    print("\nTesting ActionMemoryInferenceManager...")
    
    manager = ActionMemoryInferenceManager(
        action_dim=7,
        memory_dim=512,
        trajectory_window=10,
        max_memory_capacity=20,
        device="cpu",
    )
    
    # Simulate adding actions
    for step in range(15):
        action = torch.randn(7)
        manager.add_action(action)
    
    # Get memory
    keys, values = manager.get_memory()
    
    if keys is not None:
        assert keys.shape[0] == 1, f"Expected batch size 1, got {keys.shape[0]}"
        assert keys.shape[2] == 512, f"Expected memory_dim 512, got {keys.shape[2]}"
        print(f"  Memory shape: {keys.shape}")
        print(f"  Memory bank size: {len(manager.memory_bank)}")
    else:
        print("  Memory is empty (expected if < 10 actions)")
    
    print("✓ ActionMemoryInferenceManager passed")


def test_dit_integration():
    """Test DiT integration (if DiT is available)."""
    print("\nTesting DiT Integration...")
    
    try:
        from gr00t.model.action_head.cross_attention_dit import DiT
        
        # Create DiT with memory support
        dit = DiT(
            num_attention_heads=8,
            attention_head_dim=64,
            output_dim=7,
            num_layers=12,
            enable_memory_cross_attention=True,
            memory_cross_attention_layers=(3, 7, 10, 11),
            memory_dim=512,
        )
        
        # Test inputs
        hidden_states = torch.randn(2, 16, 512)  # (B, T, D)
        encoder_hidden_states = torch.randn(2, 32, 512)  # (B, S, D)
        timestep = torch.randint(0, 1000, (2,))
        memory_keys = torch.randn(2, 10, 512)  # (B, N, D_mem)
        memory_values = torch.randn(2, 10, 512)  # (B, N, D_mem)
        
        # Forward pass
        output = dit(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            memory_keys=memory_keys,
            memory_values=memory_values,
        )
        
        assert output.shape == (2, 16, 7), f"Expected (2, 16, 7), got {output.shape}"
        print("✓ DiT Integration passed")
        
    except ImportError as e:
        print(f"⚠ DiT not available, skipping integration test: {e}")
    except Exception as e:
        print(f"✗ DiT Integration failed: {e}")
        raise


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running Action Memory Integration Tests")
    print("="*60)
    
    test_trajectory_encoder()
    test_memory_bank()
    test_memory_cross_attention()
    test_training_wrapper()
    test_inference_manager()
    test_dit_integration()
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

