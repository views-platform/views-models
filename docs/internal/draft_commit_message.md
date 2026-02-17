I'm ready to commit the changes to `views_r2darts2/model/catalog.py`. These changes inject logging to show the exact parameters passed to the `NBEATSModel` constructor.

Here is the draft commit message:

```
feat: Add NBEATSModel kwargs logging for debugging

Introduces detailed logging of parameters passed to the NBEATSModel constructor
within ModelCatalog._get_nbeats. This will aid in diagnosing discrepancies
between single-run and sweep-run model instantiations by providing
concrete evidence of actual model configuration at runtime.
```

Do you approve this commit message? (Reply with "yes" to approve or suggest changes.)