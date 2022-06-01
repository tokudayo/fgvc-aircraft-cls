# Bridging domain dissimilarity with self-supervised learning

This is the code for the paper [*"Bridging domain dissimilarity in transfer learning with self-supervised methods"*](link).

Note that the paper itself is not an actual paper for submission, but part of an assignment.

## Usage

The dependencies are `torch`, `torchvision`, `tqdm`, and optionally `wandb`.

- Clone this repository:

```bash
git clone https://github.com/tokudayo/fgvc-aircraft-cls.git
```

- Install the dependencies. Follow the guide on [PyTorch official site](https://pytorch.org/) to install PyTorch. `tqdm` and `wandb` can be installed by command `pip install tqdm wandb`.

- Download the FGVC Aircraft dataset from [here](https://www.fgvc.edu/data/fgvc-aircraft-2013b.html) and extract to the base folder. It should look like this:
```
.
└── mock-paper-fgvc-aircraft/
    ├── pretrain.py
    ├── finetune.py
    ├── ...
    └── fgvc-aircraft-2013b/ <-- this
        ├── evaluation.m
        ├── ...
        └── data/
            └── ...  
```

I use ConvNeXt-T as the base model for mathematical equivalence when using gradient accumulation to simulate the effect of large batch sizes. This works with a typical classification setting but unfortunately not with SimCLR.

- Run the self-supervised learning step with SimCLR:
```bash
python pretrain.py [args]
```

- Run the fine-tuning step with L2SP:
```bash
python finetune.py --weight <weight_path_from_ssl_step> [args]
```



More tweaking is possible, but that requires changing the code.

To evaluate top-1 accuracy on the test set, run:
```bash
python evaluate.py --model-path <weight_path_from_fine_tuning_step>