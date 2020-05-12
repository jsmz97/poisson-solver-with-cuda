# poisson-solver-with-cuda

Access subdirectories:

- Sequential: for sequential implementation

- Parallel: for parallel implementation

```bash
cd $(subdirectory)
```

Modify parameters:

```bash
vim parameters.h
vim parameters_cu.h
```

Compile:

```bash
make
```

Build the `Outputs` folder:

```bash
make build
```

Execute:

```bash
make run
```

Clean:

```bash
make clean
```

Delete (rebuild required afterwards):

```bash
make shred
```

