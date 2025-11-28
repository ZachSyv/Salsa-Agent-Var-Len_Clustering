# Initial Frame Analysis for Motion Token Decoding

## Current Initial Frame Behavior

### How Motion is Recovered from Tokens

1. **VQ Tokens → RIC Format**:
   - `motion_tokenizer.net.forward_decoder(vq_tokens)` decodes tokens to RIC format (263-dim vector)
   - RIC format contains:
     - `data[..., 0]`: Root rotation velocity (angular velocity around Y-axis)
     - `data[..., 1:3]`: Root linear velocity (on XZ plane)
     - `data[..., 3]`: Root height (Y position)
     - `data[..., 4:]`: Relative joint positions (relative to root)

2. **RIC Format → 3D Keypoints** (`recover_from_ric`):
   - Root position is computed by integrating velocities starting from **origin (0, height, 0)**
   - Root rotation starts from **0 degrees** (no initial rotation)
   - Joint positions are relative to root, then transformed to global coordinates

### Initial Frame Details

From `recover_root_rot_pos` in `utils/motion_utils.py`:

```python
# Initial rotation: 0 (starts at frame 0)
r_rot_ang[..., 0] = 0  # First frame has no rotation
r_rot_ang[..., 1:] = rot_vel[..., :-1]  # Subsequent frames integrate velocity
r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)  # Cumulative sum

# Initial position: (0, height, 0)
r_pos = torch.zeros(...)  # Starts at origin
r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]  # Integrate velocities
r_pos = torch.cumsum(r_pos, dim=-2)  # Cumulative sum
r_pos[..., 1] = data[..., 3]  # Y position from data (height)
```

**Summary**: 
- **Initial root position**: `(0, height_from_data, 0)` - X and Z start at 0, Y from data
- **Initial root rotation**: `0 degrees` around Y-axis
- **Initial joint positions**: Relative positions from RIC data, transformed to global coordinates

## Can We Use an Initial Frame to Guide Inference?

### Current Limitations

❌ **Not directly supported**: The current system does not have a mechanism to condition on an initial frame. The RIC format stores:
- Velocities (not absolute positions)
- Relative joint positions (not absolute)
- Only root height is absolute

### Potential Solutions

#### Option 1: Modify First Frame of RIC Data (Recommended)

To condition on an initial frame, you would need to:

1. **Convert initial frame to RIC format**:
   ```python
   # Convert initial keypoints to RIC format
   initial_keypoints = ...  # (22, 3) - your initial pose
   initial_ric = convert_keypoints_to_ric(initial_keypoints)
   ```

2. **Adjust decoded RIC data**:
   - Set first frame's root position to match initial frame
   - Set first frame's root rotation to match initial frame
   - Adjust relative joint positions to match initial frame

3. **Modify `recover_from_ric`** to accept initial conditions:
   ```python
   def recover_from_ric_with_initial(data, joints_num, initial_root_pos=None, initial_root_rot=None):
       # If initial conditions provided, adjust first frame
       if initial_root_pos is not None:
           r_pos[..., 0, :] = initial_root_pos
       if initial_root_rot is not None:
           r_rot_quat[..., 0, :] = initial_root_rot
       # ... rest of recovery
   ```

#### Option 2: Prepend Initial Frame to Generated Motion

After decoding, you could:
1. Decode motion tokens as usual
2. Extract first frame from your input
3. Replace the first frame of decoded motion with your initial frame
4. Adjust subsequent frames to smoothly transition

#### Option 3: Modify Generation Process

Modify the LLM generation to condition on initial frame:
- Encode initial frame as tokens
- Include in the prompt: `"Starting from pose: <initial_frame_tokens>"`
- Model generates continuation from that pose

### Implementation Considerations

**Key Functions to Modify**:
1. `recover_from_ric` in `utils/motion_utils.py` - Add initial condition parameters
2. `decode_motion_from_vq_tokens` in `visualization/visualization_utils.py` - Pass initial frame
3. `generate` methods in `models/mllm.py` - Condition generation on initial frame

**Data Flow**:
```
Initial Frame (keypoints) 
  → Convert to RIC format 
  → Modify first frame of decoded RIC 
  → Recover to keypoints
```

## Code Locations

- **Decoding**: `utils/salsa_utils/salsa_dataloader.py` - `Motion_tokenizer.forward_decoder()`
- **Recovery**: `utils/motion_utils.py` - `recover_from_ric()`
- **Root Position**: `utils/motion_utils.py` - `recover_root_rot_pos()`
- **Demo Usage**: `demo.py` lines 164-179

## Recommendations

1. **For visualization**: You can manually replace the first frame after decoding
2. **For generation**: Modify the prompt to include initial pose information
3. **For training**: Add initial frame conditioning to the model architecture

