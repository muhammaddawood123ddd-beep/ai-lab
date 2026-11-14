import nbformat

for path in ['Lab_01_Environment_Setup.ipynb', 'lab01/lab01_setup.ipynb']:
    nb = nbformat.read(path, as_version=4)
    nb.cells[23].source = (
        '# Record your repository URL here so the instructor can find it.\n'
        'REPOSITORY_URL = "https://github.com/muhammaddawood123ddd-beep/ai-lab"\n\n'
        'print("Submitted repository:", REPOSITORY_URL)'
    )
    nbformat.write(nb, path)
    print(f'Successfully updated {path}')
