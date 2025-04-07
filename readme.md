# Plain Convolution Encryption & Chaos Synchronization (Python Implementation)

This repository contains a Python implementation of the encryption methods discussed in the paper "Plain Convolution Encryption as an Alternative to Overcoming the Limitations of Synchronization-Based Methods". It explores chaos synchronization using the Lorenz system and the proposed Plain Convolution Encryption (PCE) method (both addition and convolution variants).

The code aims to replicate the findings of the paper, demonstrating the fidelity and security aspects (particularly against Fourier analysis attacks) of these chaos-based communication techniques.

**Paper Reference:**

*   **Title:** Plain Convolution Encryption as an Alternative to Overcoming the Limitations of Synchronization-Based Methods
*   **Authors:** Flavio Rosales-Infante, M.L. Romero-Amezcua, Iván Álvarez-Rios, F. S Guzmán
*   **arXiv Link:** [https://arxiv.org/abs/2504.03027v1](https://arxiv.org/abs/2504.03027v1)

## Requirements

*   Python (>= 3.10 recommended)
*   NumPy
*   SciPy
*   pytest (for running tests)
*   Matplotlib (optional, for visualizing results similar to the paper's figures)

A `requirements.txt` file is included for easy installation of dependencies.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ramsyana/lorenz-chaos-encrypt.git
    cd YOUR_REPOSITORY_NAME
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    # venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Testing

This project uses `pytest` for unit testing. To run the tests:

1.  Make sure you are in the root directory of the project (where the `pytest.ini` or `pyproject.toml` file would be, or just the main project folder).
2.  Ensure your virtual environment is activated.
3.  Run the following command:

    ```bash
    pytest
    ```
    Or for more detailed output:
    ```bash
    pytest -v
    ```

The tests cover individual functions within the modules (`lorenz.py`, `messages.py`, `pce.py`, `hacking.py`, etc.) to ensure correctness based on the paper's algorithms and expected numerical behavior.

## License

This project is licensed under the MIT License.

```text
MIT License

Copyright (c) 2025 [Ramsyana/ramsyana.com - ramsyana[at]mac.com]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Collaboration & Contribution

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/ramsyana/lorenz-chaos-encrypt/issues).

If you'd like to contribute:

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## Contact

Ramsyana - ramsyana[at]mac.com

If you're interested in collaborating on the mathematical aspects of chaos theory, cryptography, or related mathematical research within this project, please don't hesitate to reach out. I welcome discussions on theoretical foundations, numerical methods, and mathematical optimizations.
