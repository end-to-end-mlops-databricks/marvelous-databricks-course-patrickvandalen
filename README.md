<h1 align="center">
Marvelous MLOps End-to-end MLOps with Databricks course

## Practical information
- Weekly lectures on Wednesdays 16:00-18:00 CET.
- Code for the lecture is shared before the lecture.
- Presentation and lecture materials are shared right after the lecture.
- Video of the lecture is uploaded within 24 hours after the lecture.

- Every week we set up a deliverable, and you implement it with your own dataset.
- To submit the deliverable, create a feature branch in that repository, and a PR to main branch. The code can be merged after we review & approve & CI pipeline runs successfully.
- The deliverables can be submitted with a delay (for example, lecture 1 & 2 together), but we expect you to finish all assignments for the course before the 25th of November.


## Set up your environment
In this course, we use Databricks 15.4 LTS runtime, which uses Python 3.11.
In our examples, we use UV. Check out the documentation on how to install it: https://docs.astral.sh/uv/getting-started/installation/

To create a new environment and create a lockfile, run:

```
uv venv -p 3.11.0 venv
source venv/bin/activate
uv pip install -r pyproject.toml --all-extras
uv lock
```

pre-commit run:

```
uv pip install pre-commit
uv run pre-commit run --all-files
```

copy wheel:

```
uv build
databricks fs cp dist/mlops_with_databricks-0.0.1.tar.gz dbfs:/Volumes/mdl_europe_anz_dev/patrick_mlops/mlops_course/mlops_with_databricks-0.0.1-py3-none-any.whl

```

if databricks-cli does not work anymore, then run:
pip install databricks-cli

----

deploy bundle in the bundle resource explorer or run:
databricks bundle validate
databricks bundle deploy
