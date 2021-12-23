# Crypto Trading Robot

Execute the following command to ignore changes to already-tracked connections.py to not leak private credentials:
```
git update-index --assume-unchanged connections.py
```

Execute the following command to set up a conda environment. If you encounter an error, access environment.yml and change the prefix:
```
conda env update --file environment.yml
```

Execute the following command to update the YAML file in your active directory when you modify the list of dependencies:
```
conda env export --no-builds > environment.yml
```

## Utilities
