
# Class Intent Contract: PackageScaffoldBuilder

**Status:** Active  
**Owner:** Project maintainers  
**Last reviewed:** 2026-04-05  
**Related ADRs:** ADR-001, ADR-002, ADR-006, ADR-008  

---

## 1. Purpose

`PackageScaffoldBuilder` creates the directory structure and initial files for a new model architecture package (e.g., `views-stepshifter`). It delegates package creation and validation to `views_pipeline_core.managers.package.PackageManager` and adds supplementary files (`.gitignore`, example manager script).

---

## 2. Non-Goals (Explicit Exclusions)

- Does **not** implement any algorithm logic; it generates the skeleton only
- Does **not** validate that the generated package is importable or functional
- Does **not** register the package with PyPI, GitHub, or any package index
- Does **not** create model directories — that is `ModelScaffoldBuilder`'s responsibility
- Does **not** manage conda environments or dependencies

---

## 3. Responsibilities and Guarantees

- Guarantees that `PackageManager.create_views_package()` and `PackageManager.validate_views_package()` are called in sequence
- Guarantees that errors from `create_views_package()` or `validate_views_package()` are logged at ERROR level and re-raised (fail-loud)
- Guarantees that `build_package_directories()` creates the manager subdirectory if it does not exist
- Guarantees that `build_package_scripts()` generates an `example_manager.py` from the `views_pipeline_core` template
- Guarantees that `add_gitignore()` generates a `.gitignore` from the `views_pipeline_core` template

---

## 4. Inputs and Assumptions

- **Constructor:** Receives a `PackageManager` instance (from `views_pipeline_core`)
- **Assumption:** The `PackageManager` is initialized with a valid path and `validate=False` (validation happens in `build_package_scaffold()`)
- **Assumption:** `views_pipeline_core.templates.package` templates (`template_example_manager`, `template_gitignore`) are installed and accessible

### Interactive CLI (when run as `__main__`):

- Prompts for `package_name` — validated by `PackageManager.validate_package_name()` (format: `views-packagename`, lowercase)
- Prompts for optional `package_path` — defaults to `cwd / package_name`; validated that parent directory exists

---

## 5. Outputs and Side Effects

- **Directories created:** Package root, manager subdirectory
- **Files created:** `example_manager.py`, `.gitignore`, plus whatever `PackageManager.create_views_package()` generates internally
- **Logging:** INFO-level log on successful scaffold creation; ERROR-level on failure
- **No return values:** All methods return `None`

---

## 6. Failure Modes and Loudness

| Condition | Behavior |
|---|---|
| `create_views_package()` raises | Logged at ERROR, re-raised — **fail loud** |
| `validate_views_package()` raises | Logged at ERROR, re-raised — **fail loud** |
| Invalid package name (interactive) | Loops until valid input provided |
| Invalid base directory (interactive) | Loops until valid path provided |
| Template files missing | Raises from `views_pipeline_core` — not caught here |

All structural failures propagate. No silent degradation.

---

## 7. Boundaries and Interactions

| Interacts With | Direction | Nature |
|---|---|---|
| `views_pipeline_core.managers.package.PackageManager` | Delegates to | Package creation and validation |
| `views_pipeline_core.templates.package.template_example_manager` | Calls | File generation |
| `views_pipeline_core.templates.package.template_gitignore` | Calls | File generation |

Does **not** interact with: model directories, config files, conda environments, or any other scaffold builder.

---

## 8. Examples of Correct Usage

```python
from views_pipeline_core.managers.package import PackageManager

pm = PackageManager("/path/to/views-newpackage", validate=False)
builder = PackageScaffoldBuilder(pm)
builder.build_package_scaffold()
builder.build_package_directories()
builder.build_package_scripts()
builder.add_gitignore()
```

---

## 9. Examples of Incorrect Usage

```python
# Wrong: using PackageScaffoldBuilder to create a model directory
builder = PackageScaffoldBuilder(some_package_manager)
builder.build_package_scaffold()
# This creates a *package* skeleton, not a model directory

# Wrong: calling build_package_scripts() before build_package_directories()
# The manager/ subdirectory may not exist yet
builder.build_package_scripts()  # may fail if directory missing
```

---

## 10. Test Alignment

- **No tests exist** for `PackageScaffoldBuilder`. This is a known gap (Risk Register C-07).
- A green-team test should verify: scaffold output creates expected directories and files.
- A beige-team test should verify: generated package name matches `views-*` convention.

---

## 11. Evolution Notes

- If `views_pipeline_core` templates change, scaffold output changes silently — no regression test catches this.
- The class is thin (~30 lines of logic); most complexity lives in `PackageManager`. If `PackageManager` evolves, this class may need updates.

---

## 12. Known Deviations

- **No tests:** The scaffold builder is entirely untested (Risk Register C-07). A `views_pipeline_core` template change could silently break newly scaffolded packages.
- **No assessment method:** Unlike `ModelScaffoldBuilder`, this class has no `assess_*` method to validate what it created.
- **Execution order dependency:** `build_package_scripts()` assumes `build_package_directories()` has already run, but this ordering is not enforced or documented in code.

---

## End of Contract

This document defines the **intended meaning** of `PackageScaffoldBuilder`.

Changes to behavior that violate this intent are bugs.  
Changes to intent must update this contract.
