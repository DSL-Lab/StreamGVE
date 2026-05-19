# LongLive & StreamGVE

This folder contains the StreamGVE implementation built on [LongLive v1.0](https://github.com/NVlabs/LongLive/tree/v1.0) codebase.

Most files are inherited from the upstream LongLive repository. The implementation of StreamGVE is mainly located in:

- `inference_edit_streamgve.py`: command-line entry point for StreamGVE video editing.
- `inference_edit_streamgve.sh`: example editing commands.
- `pipeline/edit_causal_inference.py`: dual-branch editing pipeline adapted to the LongLive streaming generator.
- `pipeline/__init__.py`: exposes `EditCausalInferencePipeline`.
- `wan/modules/model.py` and `wan/modules/causal_model.py`: attention-level support for grounding, boosting, query/key blending, source KV injection, and mask-aware cache usage.

The original LongLive training, inference, and interactive inference files are kept for code compatibility and attribution. For full upstream documentation, please refer to the original LongLive repository.

## Quick Run

Install dependencies from the top-level `requirements.txt`, then prepare checkpoints:

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
huggingface-cli download Efficient-Large-Model/LongLive --local-dir longlive_models
```

Run the provided editing examples:

```bash
bash inference_edit_streamgve.sh
```

For first-frame visual prompting, the example script uses `--triple_first_frame` by default to better align with LongLive's sink token design.

## Acknowledgements

This folder is based on LongLive, which is licensed under Apache-2.0. Please keep the upstream license and cite LongLive when using this implementation.
