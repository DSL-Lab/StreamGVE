# Self Forcing & StreamGVE

This folder contains the StreamGVE implementation built on [Self Forcing](https://github.com/guandeh17/Self-Forcing) codebase.

Most files are inherited from the upstream Self Forcing repository. The implementation of StreamGVE is mainly located in:

- `inference_edit_streamgve.py`: command-line entry point for StreamGVE video editing.
- `inference_edit_streamgve.sh`: example editing commands.
- `pipeline/edit_causal_inference.py`: dual-branch editing pipeline, attention bridge, grounding/boosting, source-oriented guidance, and visual prompting logic.
- `pipeline/__init__.py`: exposes `EditCausalInferencePipeline`.
- `wan/modules/model.py` and `wan/modules/causal_model.py`: attention-level support for grounding, boosting, query/key blending, source KV injection, and mask-aware cache usage.

The original Self Forcing training and generation files are kept for code compatibility and attribution. For full upstream documentation, please refer to the original Self Forcing repository.

## Quick Run

Install dependencies from the top-level `requirements.txt`, then prepare checkpoints:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir-use-symlinks False --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
python setup.py develop
```

Run the provided editing examples:

```bash
bash inference_edit_streamgve.sh
```

Refer to [Self Forcing Issue 2](https://github.com/guandeh17/Self-Forcing/issues/2), we implemented rollout-based long-video editing for Self Forcing, which is different from LongLive's natural adaptation to any length.

## Acknowledgements

This folder is based on Self Forcing, which is licensed under Apache-2.0. Please keep the upstream license and cite Self Forcing when using this implementation.
