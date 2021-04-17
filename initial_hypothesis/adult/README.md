Training scripts for adult income dataset available at: https://archive.ics.uci.edu/ml/datasets/adult.

```
ssh gpucluster.doc.ic.ac.uk
cd private-pipelines/initial-hypothesis/adult
sbatch slurm_train_models.sh
```

Each training batch of models will be saved in vol/al5217-tmp/adult/models/batch-{batch_number}