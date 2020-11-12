# PyVRC

PyVRC is a Python package to fit Variable Rate Coding models to behavioral data.


## Usage


First, make sure you have Python installed (3.6 and newer) and then use pip to install the `pyvrc` package:


```python
pip3 install pyvrc -U --index-url=https://__token__:<your_personal_token>@gitlab.uni.lu/api/v4/projects/2030/packages/pypi/simple
```

See [FAQ](#faq) section below for more information about personal access tokens.

Upon successfuly installation, all functionalities will be accessible via `vrc` module. This module exposes all the public APIs.

```python
import vrc
```

## FAQ

<details>
<summary><b>Why do I need "personal access token" to install the package?</b></summary>

Personal access tokens provide read-only access to the GitLab package registry and allow you to install PyVRC in your notebooks (e.g., Google Colab) without revealing your username/password or granting  access to private projects on GitLab.

This is temporary and whenever the package is stable, it will be accessible via public PyPI or a common project-level token.

</details>


<details>
<summary><b>How can I create a "personal access token"?</b></summary>

Personal access tokens can be created in [GitLab User Setting > Access Tokens](https://gitlab.uni.lu/profile/personal_access_tokens).

Make sure the newly created token grants read-only access to container registry images on private projects (`read_registry`).

</details>