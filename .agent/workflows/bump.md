---
description: bump the version number in setup.cfg and src/gridvoting_jax/__init__.py
---

1. Update `setup.cfg` with the new version.
```bash
sed -i 's/^version = .*/version = <version>/' setup.cfg
```

2. Update `src/gridvoting_jax/__init__.py` with the new version.
```bash
sed -i 's/^__version__ = .*/__version__ = "<version>"/' src/gridvoting_jax/__init__.py
```

3. Verify the changes
```bash
grep "version" setup.cfg src/gridvoting_jax/__init__.py
```
