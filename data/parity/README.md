# Parity reference

`idrid_demo_reference.npz` holds, for the ten images of `../demo_images/` (in sorted file-name order), the Stage-1
quality probabilities, the fundus boxes and the logits of all six graders (two frame variants, three seeds). They were
computed with float32 on a CPU (Windows, TensorFlow 2.21.0, Keras 3.15.1) by the development implementation, which this
repository's code reproduces exactly on that platform.

```bash
python scripts/parity_check.py --images data/demo_images --reference data/parity/idrid_demo_reference.npz
```

Tolerances: logits 1e-5 (a Linux or Docker CPU build differs by up to about 1.5e-5: pass
`--tolerance logits=1e-4`), quality probabilities 1e-5, fundus boxes and decoded grades identical.
