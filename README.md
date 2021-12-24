# Stock Trading Robot

Execute the following command to ignore changes to already-tracked connections.py to not leak private credentials:
```
git rm --cached connections.py
git update-index --assume-unchanged connections.py
```

Execute the following commands if you intend to use a conda environment.
The following commands require setting the project root directory as active directory.
```
cd <project-root-directory>
```

Execute the following command to create a conda environment for first-time. If you are using a non-Linux-based OS and you encounter an error, access environment.yml and create a prefix variable:
```
conda env create -f environment.yml
```

After creating the conda environment, execute the following command to activate it:
```
conda activate <new-env>
```

Execute the following command to update the existing conda environment with YAML file: 
```
conda env update -f environment.yml
```

Execute the following command to update the YAML file in your active directory when you modify the list of dependencies:
```
conda env export --no-builds > environment.yml
```

Execute the following command to undo changes in conda environment:
```
conda list --revisions
conda install --revision <revision-number>
```

Execute the following command to delete a conda environment (must not be active):
```
conda env remove -n <env-name>
```

## Utilities
