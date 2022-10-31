# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
## Challenges in adapting Transformer from language to vision
1. Large variations in the scale of visual entities. Same semantic objects may have significant different scales.
2. High resolution

## Contributions
1. The shifted windowing scheme brings greater efficiency by limiting self-attention to non-overlapping local windows while also allowing for **cross-window connection**.
2. The hierarchical architecture has the flexibility to model at **various scales** and has **linear** computational complexity.
3. Swin Transformer can serve as a general-purpose backbone.

## Algorithm
![Swin Transformer Architecture](Images/Swin_Architecture.png)
- Patch partition: partition the image to patches of $4\times 4$. So the feature dimension would be $3\times 4\times 4=48$ before linear embedding.
- Linear embedding: change the feature dimension to $C$
- Swin Transformer Block conducts self-attention only inside a **local** window which is consisted of $7\times 7$ patches. This step is different from ViT where self-attention is calculated between all patches. 
- Patch Merging:
  ![Patch Merging](Images/Swin_PatchMerging.png)
  1. Anti-flatten the sequence output from transformer back to 3d.
  2. Rearrange the input with a stride of 2. 
  3. Concatenate the new four windows along the channel dimension. 
  4. LayerNorm
  5. Convolution layer with kernel $1\times 1$ to change the channel dimension.


### Swin Transformer Block: Shift Window based Self-Attention
![Shifted Window](Images/Swin_Window.png)
- Self-attention in non-overlapped windows
  - Partition the input to $H/4\times W/4$ patches.
  - Adjacent $M\times M$ patches construct a window, thus $H/4M\times W/4M$ windows.
  - 